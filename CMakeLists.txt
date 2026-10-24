cmake_minimum_required(VERSION 3.22)
project(FreezeRoll VERSION 0.1.0)

include(FetchContent)
FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4)
FetchContent_MakeAvailable(JUCE)

# AU is only built on macOS; the format is ignored elsewhere
juce_add_plugin(FreezeRoll
    COMPANY_NAME "YourName"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    PLUGIN_MANUFACTURER_CODE Yrnm
    PLUGIN_CODE Frzr
    FORMATS VST3 AU Standalone
    PRODUCT_NAME "FreezeRoll")

target_sources(FreezeRoll PRIVATE Source/PluginProcessor.cpp)

target_compile_definitions(FreezeRoll PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0)

target_link_libraries(FreezeRoll
    PRIVATE juce::juce_audio_utils juce::juce_dsp
    PUBLIC juce::juce_recommended_config_flags
           juce::juce_recommended_lto_flags
           juce::juce_recommended_warning_flags)
