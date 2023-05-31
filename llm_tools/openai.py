def reset_openai_globals():
    import openai
    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key = None
    openai.api_version = None