from langchain_community.tools.ddg_search.tool import (
        DuckDuckGoSearchResults,
        DuckDuckGoSearchRun,
    )


search_tool=DuckDuckGoSearchRun()

result=search_tool.invoke("Impact of 2026 War on Society")

print(result)