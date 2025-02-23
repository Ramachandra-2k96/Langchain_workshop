from datetime import datetime
import json
import ollama
from llm_axe import OllamaChat
from llm_axe import OnlineAgent
from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser

model = ChatOllama(model="llama3.2:latest", keep_alive=1, format='json')

terminal_colors = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "reset": "\033[0m",
    "reset_foreground": "\033[39m",
    "reset_background": "\033[49m",
    "background": {
        "black": "\033[40m",
        "red": "\033[41m",
        "green": "\033[42m",
        "yellow": "\033[43m",
        "blue": "\033[44m",
        "magenta": "\033[45m",
        "cyan": "\033[46m",
        "white": "\033[47m"
    }
}

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    ans = a + b
    print(f"{terminal_colors['blue']}AI ==> {terminal_colors['cyan']}", ans, f"{terminal_colors['reset']}")
    return ans

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    ans = a * b
    print(f"{terminal_colors['blue']}AI ==> {terminal_colors['cyan']}", ans, f"{terminal_colors['reset']}")
    return ans

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts a and b."""
    ans = a - b
    print(f"{terminal_colors['blue']}AI ==> {terminal_colors['cyan']}", ans, f"{terminal_colors['reset']}")
    return ans

@tool
def devide(a: float, b: float) -> float:
    """Divides a and b."""
    ans = a / b
    print(f"{terminal_colors['blue']}AI ==> {terminal_colors['cyan']}", ans, f"{terminal_colors['reset']}")
    return ans

@tool
def use_Internet(query: str) -> str:
    """ Use internet if there is a shortage of information or anything related to up-to-date information"""
    model = OllamaChat(model='llama3.2:latest')
    online_model = OnlineAgent(model)
    result = online_model.search(query)
    print(f"{terminal_colors['blue']}AI ==> {terminal_colors['cyan']}", result, f"{terminal_colors['reset']}")
    return result

@tool
def normal_usecase(query: str) -> str:
    """ Use this function to reply to queries that you can answer without additional data or the need for more information from the internet and if someone asking wether you can use internet or not then you have to tell them that you can use internet to serch"""
    global messages
    # Pass messages as a list of dictionaries with role and content
    response_stream = ollama.chat(model='llama3.2:latest', messages=messages, stream=True)
    output = ''
    print(f"{terminal_colors['blue']}AI ==> {terminal_colors['cyan']}", end='')
    for chunk in response_stream:
        output += chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
    print(f"{terminal_colors['reset']}")
    return output

@tool
def date() -> str:
    """Tells today's date and time"""
    date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{terminal_colors['blue']}AI ==> {terminal_colors['cyan']}", date_time, f"{terminal_colors['reset']}")
    return date_time

render = render_text_description([add, subtract, multiply, devide, use_Internet, date, normal_usecase])

prompt = ChatPromptTemplate.from_messages([('system', f"you are an AI assistant who has the capability of using internet and as well as many tools, here are the available tools with descriptions:\n{render}\nYou are not allowed to call any other tools other than the ones listed here. Please strictly follow this rule. For the given input, return the name and input of the tool to use. Return your JSON object blob with both 'name' and 'arguments' keys, and also make sure that the returned argument is in dictionary format. Do not return anything else.If somebody asks you about what and all the things you are capable of then it is a normal usecase function"), ('user', "{input}")])
chain = prompt | model | JsonOutputParser()

def selector(response):
    # Use the invoke method and pass the arguments as a single dictionary
    return globals()[response['name']].invoke(response['arguments'])

messages = []
while True:
    try:
        query = input("User ==> ")
        if query.lower() =='exit':
            print(f"{terminal_colors['blue']}AI ==> {terminal_colors['red']}","Exiting...\n BYE ", f"{terminal_colors['reset']}")
            break
        stream = chain.invoke({"input": query})
        json_q = json.dumps(stream, indent=3)
        print(f"\n{terminal_colors['magenta']}DEBUG :\n{terminal_colors['green']}{json_q}{terminal_colors['reset']}\n\n")
        
        # Create HumanMessage and append as a dictionary to the messages list
        human_message = {'role': 'user', 'content': query}
        messages.append(human_message)
        
        # Process the response using the correct tool
        response = selector(stream)
        
        # Create AIMessage and append as a dictionary to the messages list
        ai_message = {'role': 'assistant', 'content': str(response)}
        messages.append(ai_message)
        
    except Exception as e:
        print(f"Error: {e}")
