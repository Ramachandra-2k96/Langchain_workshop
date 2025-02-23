# Explanation of the Web Automation Code

## Overview
This script automates web interactions using Playwright, an automation framework for browser interactions. It uses AI to determine actions like clicking, typing, scrolling, and navigating web pages. The script is designed to help an AI agent interact with web pages in a structured and logical way, following specific rules and constraints.

---

## Dependencies
The script imports several Python libraries to function properly:
- `os` and `dotenv`: Used for loading environment variables (API keys, etc.).
- `asyncio`: Enables asynchronous operations.
- `platform`: Helps detect the operating system (useful for keyboard shortcuts).
- `playwright.async_api`: Provides browser automation capabilities.
- `langchain_core`: Helps in structuring AI-generated responses.
- `langchain_groq`: Provides AI model access for generating responses.
- `re`: Used for processing text and extracting information.

---

## Key Components

### 1. **Setting Up Environment Variables**
```python
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
```
This loads environment variables from a `.env` file to securely access API keys.

### 2. **Handling Jupyter Notebook Compatibility**
```python
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass
```
If the script is run inside a Jupyter notebook, `nest_asyncio` allows running asynchronous operations properly.

### 3. **Defining Data Structures**
- **Bounding Box (BBox)**: Represents elements on a webpage with attributes like position (`x`, `y`), text, type, and `ariaLabel`.
- **Prediction**: Represents an action (like clicking or typing) along with its arguments.
- **Agent State**: Stores all relevant information about the agent's current state.

```python
from typing import List, Optional, TypedDict

class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict):
    page: "Page"  # Represents the browser page
    input: str  # User input request
    img: str  # Base64 screenshot of the page
    bboxes: List[BBox]  # List of identified elements on the page
    prediction: Prediction  # AI-generated action
    scratchpad: List[str]  # Stores intermediate steps
    observation: str  # Stores feedback from executed actions
```

---

## Action Functions
These functions execute specific actions based on the AI's predictions.

### 1. **Click**
Finds an element by its label and clicks on it.
```python
async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"].get("args", [])
    if len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = int(click_args[0])
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"
```

### 2. **Type Text**
Finds an input field and types text into it.
```python
async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"].get("args", [])
    if len(type_args) != 2:
        return "Failed to type text."
    bbox_id, text_content = int(type_args[0]), type_args[1]
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    await page.keyboard.type(text_content)
    return f"Typed '{text_content}' and submitted."
```

### 3. **Scroll**
Scrolls the page or a specific element up or down.
```python
async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"].get("args", [])
    if len(scroll_args) != 2:
        return "Failed to scroll."
    target, direction = scroll_args
    scroll_amount = 500 if target.upper() == "WINDOW" else 200
    scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
    if target.upper() == "WINDOW":
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    return f"Scrolled {direction}"
```

### 4. **Other Actions**
- **Wait**: Pauses execution for a few seconds.
- **Go Back**: Navigates back to the previous page.
- **Go to Google**: Redirects the page to Google.
- **Answer**: Provides the final response when the AI has completed its task.

---

## AI Decision-Making
The AI receives observations about the webpage and decides the next step using a structured prompt:
```python
prompt = """
Imagine you are a robot browsing the web, just like humans. You can ONLY perform ONE specific action at a time:
1. Click [Numerical_Label]
2. Type [Numerical_Label]; [Content]
3. Scroll [Numerical_Label or WINDOW]; [up or down]
4. Wait
5. GoBack
6. Google
7. ANSWER [content]
...
Your response must ALWAYS follow this format:
Thought: {{Brief analysis}}
Action: {{ONE action}}
"""
```
The AI then processes the prompt and decides on the next action.

---

## Running the Browser
The script launches a browser session using Playwright and executes actions based on the AI's responses.
```python
from playwright.async_api import async_playwright

async def run_browser():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.google.com")
        response = await call_agent("search any one interesting topic you want and have fun.")
        print(f"Final Response: {response}")
        await browser.close()
```

### **Key Features of This Setup:**
- The browser operates **without headless mode**, meaning it is visible.
- It prevents detection by disabling automation features.
- The AI can make **up to 150 decisions per session**.

---

## Conclusion
This script allows an AI agent to navigate the web autonomously by clicking, typing, scrolling, and searching information. The AI follows strict guidelines to make decisions step by step. With further improvements, this setup could be expanded to handle more complex browsing tasks, including filling out forms, solving CAPTCHAs, and summarizing web content.

