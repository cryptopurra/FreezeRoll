// FreezeRoll - spectral freeze (TimeFreeze-style) + one-knob bar-locked roll-up (Stucker-style)
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>

using namespace juce;

namespace ids
{
    constexpr const char* freeze    = "freeze";
    constexpr const char* freezeMix = "freezeMix";
    constexpr const char* order     = "order";
    constexpr const char* amount    = "amount";
    constexpr const char* grid      = "grid";
    constexpr const char* sync      = "sync";
    constexpr const char* smooth    = "smooth";
    constexpr const char* lowCut    = "lowCut";
    constexpr const char* range     = "range";
}

//==============================================================================
// Spectral freeze: grab last N samples, keep magnitudes, resynthesise with random phase.
struct FreezeEngine
{
    static constexpr int fftOrder = 13, N = 1 << fftOrder, hop = N / 4;
    static constexpr float compensation = 1.33f; // tune by ear / meter

    struct Ch
    {
        std::vector<float> ring, acc, mag, fftData;
        int w = 0, outPos = 0;
        Random rng;
    };

    dsp::FFT fft { fftOrder };
    std::vector<float> window;
    Ch ch[2];
    bool frozen = false, idle = true;
    float gain = 0.f, gainStep = 0.001f;
    int hopCount = 0;

    void prepare (double sr)
    {
        window.resize (N);
        for (int i = 0; i < N; ++i)
            window[(size_t) i] = 0.5f - 0.5f * std::cos (MathConstants<float>::twoPi * (float) i / (float) N);

        for (auto& c : ch)
        {
            c.ring.assign ((size_t) N, 0.f);
            c.acc.assign ((size_t) N, 0.f);
            c.mag.assign ((size_t) N / 2 + 1, 0.f);
            c.fftData.assign ((size_t) N * 2, 0.f);
            c.w = c.outPos = 0;
            c.rng.setSeedRandomly();
        }
        gainStep = 1.f / (float) (0.05 * sr);
        gain = 0.f; hopCount = 0; frozen = false; idle = true;
    }

    void setFrozen (bool f)
    {
        if (f && ! frozen) capture();
        frozen = f;
    }

    void capture()
    {
        for (auto& c : ch)
        {
            for (int i = 0; i < N; ++i)
                c.fftData[(size_t) i] = c.ring[(size_t) ((c.w + i) & (N - 1))] * window[(size_t) i];
            std::fill (c.fftData.begin() + N, c.fftData.end(), 0.f);
            fft.performRealOnlyForwardTransform (c.fftData.data());
            for (int k = 0; k <= N / 2; ++k)
                c.mag[(size_t) k] = std::hypot (c.fftData[(size_t) (2 * k)], c.fftData[(size_t) (2 * k + 1)]);
            std::fill (c.acc.begin(), c.acc.end(), 0.f);
            c.outPos = 0;
        }
        hopCount = hop - 1;
    }

    void synth (Ch& c)
    {
        auto* d = c.fftData.data();
        for (int k = 0; k <= N / 2; ++k)
        {
            const float ph = c.rng.nextFloat() * MathConstants<float>::twoPi;
            d[2 * k]     = c.mag[(size_t) k] * std::cos (ph);
            d[2 * k + 1] = c.mag[(size_t) k] * std::sin (ph);
        }
        d[1] = 0.f; d[N + 1] = 0.f;
        fft.performRealOnlyInverseTransform (d);
        for (int i = 0; i < N; ++i)
            c.acc[(size_t) ((c.outPos + i) & (N - 1))] += d[i] * window[(size_t) i] * compensation;
    }

    void process (float* x, int nch, float mix)
    {
        for (int c = 0; c < nch; ++c)
        {
            ch[c].ring[(size_t) ch[c].w] = x[c];
            ch[c].w = (ch[c].w + 1) & (N - 1);
        }

        gain += jlimit (-gainStep, gainStep, (frozen ? 1.f : 0.f) - gain);

        if (gain <= 0.f && ! frozen)
        {
            if (! idle) { for (auto& c : ch) std::fill (c.acc.begin(), c.acc.end(), 0.f); idle = true; }
            return;
        }
        idle = false;

        float wet[2] = { 0.f, 0.f };
        for (int c = 0; c < nch; ++c)
        {
            auto& k = ch[c];
            wet[c] = k.acc[(size_t) k.outPos];
            k.acc[(size_t) k.outPos] = 0.f;
            k.outPos = (k.outPos + 1) & (N - 1);
        }
        if (++hopCount >= hop)
        {
            hopCount = 0;
            for (int c = 0; c < nch; ++c) synth (ch[c]);
        }

        const float g = gain * mix;
        for (int c = 0; c < nch; ++c)
            x[c] = x[c] * (1.f - g) + wet[c] * g;
    }
};

//==============================================================================
// Bar-locked roll-up: capture previous grid cell, then halve its length as AMOUNT rises.
struct RollEngine
{
    enum State { Idle, Waiting, Active };

    struct Params
    {
        bool request = false, sync = true;
        float amount = 0.f, range = 8.f, smooth = 0.35f, hpA = 1.f;
        double gridBeats = 1.0, spb = 24000.0, ppq = 0.0;
    };

    std::vector<float> ring[2], cap[2];
    int w = 0, maxLen = 0, capLen = 0;
    State state = Idle;
    double phase = 0.0, lastCell = -1e9;
    float wetGain = 0.f, gainStep = 0.01f;
    float hpX[2] {}, hpY[2] {};

    void prepare (double sr)
    {
        maxLen = (int) (sr * 8.0) + 1;
        for (int c = 0; c < 2; ++c) { ring[c].assign ((size_t) maxLen, 0.f); cap[c].assign ((size_t) maxLen, 0.f); }
        w = 0; state = Idle; wetGain = 0.f; phase = 0.0; lastCell = -1e9;
        gainStep = 1.f / (float) (0.003 * sr);
        for (int c = 0; c < 2; ++c) hpX[c] = hpY[c] = 0.f;
    }

    void process (float* x, int nch, const Params& p)
    {
        for (int c = 0; c < nch; ++c) ring[c][(size_t) w] = x[c];
        w = (w + 1) % maxLen;

        const double cell = std::floor (p.ppq / p.gridBeats);
        const bool crossed = (lastCell > -1e8) && cell != lastCell;
        lastCell = cell;

        const int gridLen = jlimit (16, maxLen - 1, (int) std::llround (p.gridBeats * p.spb));

        if (! p.request && state == Waiting) state = Idle;
        else if (p.request && state == Idle)  state = Waiting;

        if (state == Waiting && (crossed || ! p.sync))
        {
            for (int c = 0; c < nch; ++c)
                for (int i = 0; i < gridLen; ++i)
                    cap[c][(size_t) i] = ring[c][(size_t) ((w - gridLen + i + maxLen) % maxLen)];
            capLen = gridLen;
            phase = 0.0;
            state = Active;
        }

        if (state != Active) return;

        wetGain += jlimit (-gainStep, gainStep, (p.request ? 1.f : 0.f) - wetGain);
        if (! p.request && wetGain <= 0.f) { state = Idle; return; }

        // halving: smooth=0 -> steps of exactly 1/2, 1/4, 1/8...; smooth=1 -> continuous glide
        const float o = p.amount * p.range, fl = std::floor (o), fr = o - fl;
        const float eff = fl + jlimit (0.f, 1.f, fr / jmax (p.smooth, 0.02f));
        const double len = jmax (8.0, (double) capLen * std::pow (2.0, (double) -eff));

        phase += 1.0;
        if (phase >= len) phase = std::fmod (phase, len);

        const int i0 = (int) phase;
        const int i1 = jmin (i0 + 1, capLen - 1);
        const float f = (float) (phase - i0);
        const double xf = jmax (1.0, jmin (64.0, len / 8.0));
        const float env = (float) jmin (1.0, jmin (phase / xf, (len - phase) / xf));

        for (int c = 0; c < nch; ++c)
        {
            const float s = (cap[c][(size_t) i0] * (1.f - f) + cap[c][(size_t) i1] * f) * env;
            const float y = p.hpA * (hpY[c] + s - hpX[c]);
            hpX[c] = s; hpY[c] = y;
            x[c] = x[c] * (1.f - wetGain) + y * wetGain;
        }
    }
};

//==============================================================================
//==============================================================================
// GUI: hardware-style panel. Big orange ROLL knob (one-knob Stucker idea) + big ice-blue FREEZE pad.
namespace col
{
    const Colour bg0 { 0xff101216 }, bg1 { 0xff07080a }, panel { 0xff16191e }, edge { 0xff2a2e35 },
                 orange { 0xffff5a1f }, ice { 0xff4fd8ff }, text { 0xffcfd3da }, dim { 0xff7d838d };
}

struct HwLook : LookAndFeel_V4
{
    HwLook()
    {
        setColour (Slider::textBoxTextColourId, col::text);
        setColour (Slider::textBoxOutlineColourId, Colours::transparentBlack);
        setColour (Label::textColourId, col::dim);
        setColour (ComboBox::backgroundColourId, col::panel);
        setColour (ComboBox::textColourId, col::text);
        setColour (ComboBox::outlineColourId, col::edge);
        setColour (ComboBox::arrowColourId, col::orange);
        setColour (PopupMenu::backgroundColourId, col::panel);
        setColour (PopupMenu::textColourId, col::text);
        setColour (PopupMenu::highlightedBackgroundColourId, col::orange);
        setColour (ToggleButton::textColourId, col::text);
        setColour (ToggleButton::tickColourId, col::orange);
        setColour (ToggleButton::tickDisabledColourId, col::dim);
    }

    void drawRotarySlider (Graphics& g, int x, int y, int w, int h, float pos,
                           float a0, float a1, Slider& s) override
    {
        const bool big = (bool) s.getProperties()["big"];
        auto b = Rectangle<float> ((float) x, (float) y, (float) w, (float) h).reduced (4.f);
        const float r = jmin (b.getWidth(), b.getHeight()) * 0.5f;
        const auto c = b.getCentre();
        const auto accent = big ? col::orange : col::text;
        const float thick = big ? 9.f : 4.f;
        const float ang = a0 + pos * (a1 - a0);

        Path track;
        track.addCentredArc (c.x, c.y, r - thick, r - thick, 0.f, a0, a1, true);
        g.setColour (Colour (0xff23262c));
        g.strokePath (track, PathStrokeType (thick, PathStrokeType::curved, PathStrokeType::rounded));

        Path val;
        val.addCentredArc (c.x, c.y, r - thick, r - thick, 0.f, a0, ang, true);
        if (big)
        {
            g.setColour (accent.withAlpha (0.25f));
            g.strokePath (val, PathStrokeType (thick + 8.f, PathStrokeType::curved, PathStrokeType::rounded));
        }
        g.setColour (accent);
        g.strokePath (val, PathStrokeType (thick, PathStrokeType::curved, PathStrokeType::rounded));

        const float kr = r * 0.68f;
        g.setGradientFill (ColourGradient (Colour (0xff3d414a), c.x, c.y - kr, Colour (0xff121418), c.x, c.y + kr, false));
        g.fillEllipse (c.x - kr, c.y - kr, kr * 2.f, kr * 2.f);
        g.setColour (Colour (0xff050607));
        g.drawEllipse (c.x - kr, c.y - kr, kr * 2.f, kr * 2.f, 2.f);

        Path p;
        p.addRoundedRectangle (-2.5f, -kr + 5.f, 5.f, kr * 0.38f, 2.f);
        g.setColour (accent);
        g.fillPath (p, AffineTransform::rotation (ang).translated (c.x, c.y));
    }
};

struct FreezeButton : ToggleButton, Timer
{
    float glow = 0.f;
    FreezeButton() { setButtonText ("FREEZE"); startTimerHz (30); }
    void timerCallback() override
    {
        glow += ((getToggleState() ? 1.f : 0.f) - glow) * 0.25f;
        repaint();
    }
    void paintButton (Graphics& g, bool over, bool) override
    {
        auto b = getLocalBounds().toFloat().reduced (6.f);
        if (glow > 0.01f)
        {
            g.setColour (col::ice.withAlpha (0.20f * glow));
            g.fillRoundedRectangle (b.expanded (5.f), 18.f);
        }
        g.setColour (Colour (0xff1a1d23).interpolatedWith (col::ice.darker (0.55f), glow));
        g.fillRoundedRectangle (b, 14.f);
        g.setColour (col::ice.withAlpha ((over ? 0.55f : 0.35f) + 0.65f * glow));
        g.drawRoundedRectangle (b, 14.f, 2.f);

        // snowflake-ish asterisk
        auto c = b.getCentre().translated (0.f, -10.f);
        g.setColour (Colours::white.interpolatedWith (col::ice, 1.f - glow * 0.5f).withAlpha (0.6f + 0.4f * glow));
        for (int i = 0; i < 3; ++i)
        {
            const float a = MathConstants<float>::pi * (float) i / 3.f;
            g.drawLine (Line<float> (c.getPointOnCircumference (16.f, a),
                                     c.getPointOnCircumference (16.f, a + MathConstants<float>::pi)), 2.f);
        }

        g.setFont (Font (FontOptions (17.f, Font::bold)));
        g.drawText (getToggleState() ? "FROZEN" : "FREEZE", b.withTrimmedTop (b.getHeight() * 0.62f).toNearestInt(),
                    Justification::centredTop);
    }
};

// Shows how the captured bar gets sliced by the roll, using the same maths as the DSP.
struct RollDisplay : Component, Timer
{
    std::atomic<float> *amountP, *rangeP, *smoothP, *freezeP;
    explicit RollDisplay (AudioProcessorValueTreeState& a)
        : amountP (a.getRawParameterValue (ids::amount)), rangeP (a.getRawParameterValue (ids::range)),
          smoothP (a.getRawParameterValue (ids::smooth)), freezeP (a.getRawParameterValue (ids::freeze))
    { startTimerHz (30); }
    void timerCallback() override { repaint(); }

    void paint (Graphics& g) override
    {
        auto b = getLocalBounds().toFloat();
        g.setColour (Colour (0xff0a0b0d));  g.fillRoundedRectangle (b, 8.f);
        g.setColour (col::edge);            g.drawRoundedRectangle (b.reduced (0.5f), 8.f, 1.f);

        auto a = b.reduced (10.f, 8.f);
        auto textArea = a.removeFromBottom (16.f);
        const float amount = amountP->load(), range = rangeP->load(), smooth = smoothP->load();
        const bool frozen = freezeP->load() > 0.5f;

        const float o = amount * range, fl = std::floor (o), fr = o - fl;
        const float eff = fl + jlimit (0.f, 1.f, fr / jmax (smooth, 0.02f));
        const bool dry = amount < 0.001f;
        const float slices = dry ? 1.f : std::pow (2.f, eff);
        const float sw = a.getWidth() / slices;

        g.setColour (col::orange.withAlpha (dry ? 0.22f : 0.9f));
        if (sw < 3.f)
        {
            g.setGradientFill (ColourGradient (col::orange.withAlpha (0.35f), a.getX(), 0.f,
                                               col::orange, a.getRight(), 0.f, false));
            g.fillRoundedRectangle (a, 2.f);
        }
        else
        {
            for (float x = a.getX(); x < a.getRight() - 0.5f; x += sw)
                g.fillRoundedRectangle (x, a.getY(), jmax (1.f, jmin (sw - 2.f, a.getRight() - x)), a.getHeight(), 2.f);
        }

        g.setFont (Font (FontOptions (11.f, Font::bold)));
        g.setColour (col::text);
        g.drawText (dry ? "DRY" : "SLICE 1/" + String (roundToInt (slices)), textArea.toNearestInt(), Justification::centredLeft);
        if (frozen)
        {
            g.setColour (col::ice);
            g.drawText ("FROZEN", textArea.toNearestInt(), Justification::centredRight);
            g.fillRect (b.getX() + 12.f, b.getBottom() - 4.f, b.getWidth() - 24.f, 2.f);
        }
    }
};

struct Knob : Component
{
    enum Fmt { Percent, Hz, Octaves };

    Slider s { Slider::RotaryHorizontalVerticalDrag, Slider::TextBoxBelow };
    Label name;
    std::unique_ptr<AudioProcessorValueTreeState::SliderAttachment> att;

    Knob (AudioProcessorValueTreeState& a, const char* id, const String& label, Fmt fmt, bool big = false)
    {
        s.setTextBoxStyle (Slider::TextBoxBelow, false, 80, 18);
        s.setRotaryParameters (MathConstants<float>::pi * 1.2f, MathConstants<float>::pi * 2.8f, true);
        s.getProperties().set ("big", big);
        att = std::make_unique<AudioProcessorValueTreeState::SliderAttachment> (a, id, s);

        if (fmt == Percent)
        {
            s.textFromValueFunction = [] (double v) { return String (roundToInt (v * 100.0)) + " %"; };
            s.valueFromTextFunction = [] (const String& t) { return t.getDoubleValue() / 100.0; };
        }
        else if (fmt == Hz)
            s.textFromValueFunction = [] (double v) { return String (roundToInt (v)) + " Hz"; };
        else
            s.textFromValueFunction = [] (double v) { return String (v, 1) + " oct"; };
        s.updateText();

        name.setText (label, dontSendNotification);
        name.setJustificationType (Justification::centred);
        name.setFont (Font (FontOptions (big ? 15.f : 11.f, Font::bold)));
        name.setColour (Label::textColourId, big ? col::orange : col::dim);
        addAndMakeVisible (s);
        addAndMakeVisible (name);
    }
    void resized() override
    {
        auto b = getLocalBounds();
        name.setBounds (b.removeFromTop (18));
        s.setBounds (b);
    }
};

// Fixed 560x470 design; the editor below scales it to whatever size the user drags to.
class FreezeRollPanel : public Component
{
public:
    explicit FreezeRollPanel (AudioProcessorValueTreeState& a)
        : display (a),
          amount (a, ids::amount,    "ROLL",       Knob::Percent, true),
          mix    (a, ids::freezeMix, "FREEZE MIX", Knob::Percent),
          smooth (a, ids::smooth,    "SMOOTH",     Knob::Percent),
          lowCut (a, ids::lowCut,    "LOW CUT",    Knob::Hz),
          range  (a, ids::range,     "RANGE",      Knob::Octaves)
    {
        setLookAndFeel (&look);
        setSize (560, 470);

        for (auto* c : std::initializer_list<Component*> { &display, &freezeBtn, &amount, &mix, &smooth, &lowCut, &range,
                                                           &gridBox, &orderBox, &syncBtn, &gridLbl, &orderLbl })
            addAndMakeVisible (c);

        auto fill = [&a] (ComboBox& box, const char* id)
        {
            if (auto* cp = dynamic_cast<AudioParameterChoice*> (a.getParameter (id)))
                box.addItemList (cp->choices, 1);
        };
        fill (gridBox, ids::grid);
        fill (orderBox, ids::order);

        syncBtn.setButtonText ("BAR SYNC");
        for (auto* l : { &gridLbl, &orderLbl })
        {
            l->setFont (Font (FontOptions (11.f, Font::bold)));
            l->setJustificationType (Justification::centredLeft);
        }
        gridLbl.setText ("ROLL GRID", dontSendNotification);
        orderLbl.setText ("ROUTING", dontSendNotification);

        freezeAtt = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>   (a, ids::freeze, freezeBtn);
        syncAtt   = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>   (a, ids::sync, syncBtn);
        gridAtt   = std::make_unique<AudioProcessorValueTreeState::ComboBoxAttachment> (a, ids::grid, gridBox);
        orderAtt  = std::make_unique<AudioProcessorValueTreeState::ComboBoxAttachment> (a, ids::order, orderBox);
    }

    ~FreezeRollPanel() override { setLookAndFeel (nullptr); }

    void paint (Graphics& g) override
    {
        g.setGradientFill (ColourGradient (col::bg0, 0.f, 0.f, col::bg1, 0.f, (float) getHeight(), false));
        g.fillAll();

        auto b = getLocalBounds().reduced (14);
        auto head = b.removeFromTop (40);
        g.setFont (Font (FontOptions (26.f, Font::bold)));
        g.setColour (col::orange);
        g.drawText ("FREEZE", head.removeFromLeft (92), Justification::centredLeft);
        g.setColour (col::ice);
        g.drawText ("ROLL", head.removeFromLeft (70), Justification::centredLeft);
        g.setFont (Font (FontOptions (11.f, Font::bold)));
        g.setColour (col::dim);
        g.drawText ("SPECTRAL FREEZE + BAR-LOCKED ROLL-UP", head, Justification::centredRight);

        g.setColour (col::panel);
        g.fillRoundedRectangle (midPanel.toFloat(), 10.f);
        g.fillRoundedRectangle (botPanel.toFloat(), 10.f);
        g.setColour (col::edge);
        g.drawRoundedRectangle (midPanel.toFloat().reduced (0.5f), 10.f, 1.f);
        g.drawRoundedRectangle (botPanel.toFloat().reduced (0.5f), 10.f, 1.f);
    }

    void resized() override
    {
        auto b = getLocalBounds().reduced (14);
        b.removeFromTop (40);
        display.setBounds (b.removeFromTop (64));
        b.removeFromTop (10);

        botPanel = b.removeFromBottom (118);
        b.removeFromBottom (10);
        midPanel = b;

        auto bot = botPanel.reduced (6);
        const int w = bot.getWidth() / 4;
        for (auto* k : { &mix, &smooth, &lowCut, &range })
            k->setBounds (bot.removeFromLeft (w));

        auto mid = midPanel.reduced (8);
        auto left = mid.removeFromLeft (140);
        auto right = mid.removeFromRight (130);
        freezeBtn.setBounds (left.reduced (2, 10));
        amount.setBounds (mid);

        auto r = right.reduced (2, 10);
        gridLbl.setBounds (r.removeFromTop (16));
        gridBox.setBounds (r.removeFromTop (28));
        r.removeFromTop (12);
        orderLbl.setBounds (r.removeFromTop (16));
        orderBox.setBounds (r.removeFromTop (28));
        r.removeFromTop (12);
        syncBtn.setBounds (r.removeFromTop (28));
    }

private:
    HwLook look;
    RollDisplay display;
    FreezeButton freezeBtn;
    Knob amount, mix, smooth, lowCut, range;
    ComboBox gridBox, orderBox;
    ToggleButton syncBtn;
    Label gridLbl, orderLbl;
    Rectangle<int> midPanel, botPanel;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> freezeAtt, syncAtt;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> gridAtt, orderAtt;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FreezeRollPanel)
};

class FreezeRollEditor : public AudioProcessorEditor
{
public:
    FreezeRollEditor (AudioProcessor& p, AudioProcessorValueTreeState& a)
        : AudioProcessorEditor (p), panel (a)
    {
        addAndMakeVisible (panel);
        setResizeLimits (minW, 353, maxW, 1175);
        getConstrainer()->setFixedAspectRatio ((double) baseW / (double) baseH);
        setResizable (true, true);

        // size comes from a separate settings file, not from the plugin state / presets
        const int w = jlimit (minW, maxW, PropertiesFile (settingsOptions()).getIntValue ("editorWidth", baseW));
        setSize (w, roundToInt ((double) w * baseH / baseW));
    }

    ~FreezeRollEditor() override
    {
        PropertiesFile pf (settingsOptions());
        pf.setValue ("editorWidth", getWidth());
        pf.saveIfNeeded();
    }

    void resized() override
    {
        panel.setTransform (AffineTransform::scale ((float) getWidth() / (float) baseW));
        panel.setBounds (0, 0, baseW, baseH);
    }

private:
    static constexpr int baseW = 560, baseH = 470, minW = 420, maxW = 1400;

    // Windows: %APPDATA%/FreezeRoll/FreezeRoll.settings
    // macOS:   ~/Library/Application Support/FreezeRoll/FreezeRoll.settings
    static PropertiesFile::Options settingsOptions()
    {
        PropertiesFile::Options o;
        o.applicationName     = "FreezeRoll";
        o.filenameSuffix      = ".settings";
        o.folderName          = "FreezeRoll";
        o.osxLibrarySubFolder = "Application Support";
        o.storageFormat       = PropertiesFile::storeAsXML;
        return o;
    }

    FreezeRollPanel panel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FreezeRollEditor)
};

//==============================================================================
class FreezeRollProcessor : public AudioProcessor
{
public:
    FreezeRollProcessor()
        : AudioProcessor (BusesProperties().withInput  ("Input",  AudioChannelSet::stereo(), true)
                                           .withOutput ("Output", AudioChannelSet::stereo(), true)),
          apvts (*this, nullptr, "STATE", createLayout())
    {}

    const String getName() const override { return "FreezeRoll"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const String getProgramName (int) override { return {}; }
    void changeProgramName (int, const String&) override {}
    bool hasEditor() const override { return true; }
    AudioProcessorEditor* createEditor() override { return new FreezeRollEditor (*this, apvts); }

    bool isBusesLayoutSupported (const BusesLayout& l) const override
    {
        return l.getMainOutputChannelSet() == l.getMainInputChannelSet()
            && (l.getMainOutputChannelSet() == AudioChannelSet::mono()
             || l.getMainOutputChannelSet() == AudioChannelSet::stereo());
    }

    void prepareToPlay (double sampleRate, int) override
    {
        sr = sampleRate;
        freezeEng.prepare (sr);
        rollEng.prepare (sr);
        internalPpq = 0.0;
    }
    void releaseResources() override {}

    void processBlock (AudioBuffer<float>& buf, MidiBuffer&) override
    {
        ScopedNoDenormals noDenormals;
        const int nch = jmin (2, buf.getNumChannels());
        const int n = buf.getNumSamples();
        if (nch == 0) return;

        double bpm = 120.0, hostPpq = 0.0;
        bool havePpq = false, playing = false;
        if (auto* ph = getPlayHead())
            if (auto pos = ph->getPosition())
            {
                if (auto b = pos->getBpm()) bpm = *b;
                if (auto p = pos->getPpqPosition()) { hostPpq = *p; havePpq = true; }
                playing = pos->getIsPlaying();
            }
        const bool useHost = havePpq && playing;
        const double spb = sr * 60.0 / jmax (20.0, bpm);
        const double ppqStart = useHost ? hostPpq : internalPpq;

        auto get = [this] (const char* id) { return apvts.getRawParameterValue (id)->load(); };

        static const double gridTable[] = { 4.0, 2.0, 1.0, 0.5, 0.25 };
        RollEngine::Params rp;
        rp.amount    = get (ids::amount);
        rp.request   = rp.amount > 0.001f;
        rp.range     = get (ids::range);
        rp.smooth    = get (ids::smooth);
        rp.sync      = get (ids::sync) > 0.5f;
        rp.gridBeats = gridTable[jlimit (0, 4, (int) get (ids::grid))];
        rp.spb       = spb;
        rp.hpA       = 1.f / (1.f + MathConstants<float>::twoPi * get (ids::lowCut) / (float) sr);

        const float fMix = get (ids::freezeMix);
        const bool freezeFirst = (int) get (ids::order) == 0;
        freezeEng.setFrozen (get (ids::freeze) > 0.5f);

        auto* d0 = buf.getWritePointer (0);
        auto* d1 = nch > 1 ? buf.getWritePointer (1) : nullptr;

        for (int i = 0; i < n; ++i)
        {
            float x[2] = { d0[i], d1 != nullptr ? d1[i] : 0.f };
            rp.ppq = ppqStart + (double) i / spb;

            if (freezeFirst) { freezeEng.process (x, nch, fMix); rollEng.process (x, nch, rp); }
            else             { rollEng.process (x, nch, rp);     freezeEng.process (x, nch, fMix); }

            d0[i] = x[0];
            if (d1 != nullptr) d1[i] = x[1];
        }

        if (! useHost) internalPpq += (double) n / spb;
    }

    void getStateInformation (MemoryBlock& dest) override
    {
        if (auto xml = apvts.copyState().createXml()) copyXmlToBinary (*xml, dest);
    }
    void setStateInformation (const void* data, int size) override
    {
        if (auto xml = getXmlFromBinary (data, size))
            if (xml->hasTagName (apvts.state.getType()))
                apvts.replaceState (ValueTree::fromXml (*xml));
    }

private:
    static AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        std::vector<std::unique_ptr<RangedAudioParameter>> p;
        p.push_back (std::make_unique<AudioParameterBool>   (ParameterID { ids::freeze, 1 },    "Freeze", false));
        p.push_back (std::make_unique<AudioParameterFloat>  (ParameterID { ids::freezeMix, 1 }, "Freeze Mix", 0.f, 1.f, 1.f));
        p.push_back (std::make_unique<AudioParameterChoice> (ParameterID { ids::order, 1 },     "Order",
                     StringArray { "Freeze > Roll", "Roll > Freeze" }, 0));
        p.push_back (std::make_unique<AudioParameterFloat>  (ParameterID { ids::amount, 1 },    "Roll Amount", 0.f, 1.f, 0.f));
        p.push_back (std::make_unique<AudioParameterChoice> (ParameterID { ids::grid, 1 },      "Roll Grid",
                     StringArray { "1 Bar", "1/2", "1/4", "1/8", "1/16" }, 2));
        p.push_back (std::make_unique<AudioParameterBool>   (ParameterID { ids::sync, 1 },      "Roll Sync", true));
        p.push_back (std::make_unique<AudioParameterFloat>  (ParameterID { ids::smooth, 1 },    "Roll Smooth", 0.f, 1.f, 0.35f));
        p.push_back (std::make_unique<AudioParameterFloat>  (ParameterID { ids::lowCut, 1 },    "Roll Low Cut",
                     NormalisableRange<float> (20.f, 500.f, 0.f, 0.5f), 80.f));
        p.push_back (std::make_unique<AudioParameterFloat>  (ParameterID { ids::range, 1 },     "Roll Range (oct)", 1.f, 8.f, 8.f));
        return { p.begin(), p.end() };
    }

    AudioProcessorValueTreeState apvts;
    FreezeEngine freezeEng;
    RollEngine rollEng;
    double sr = 44100.0, internalPpq = 0.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FreezeRollProcessor)
};

AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new FreezeRollProcessor(); }
