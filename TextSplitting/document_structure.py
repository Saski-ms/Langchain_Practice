from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyPDFLoader

splitter=RecursiveCharacterTextSplitter.from_language(
    chunk_size=100,
    chunk_overlap=0,
    language=Language.PYTHON
)


text="""
# If-else condition
my_shirt_color_today = "red"
if my_shirt_color_today == "red":
    print("Hello friend, I like the color of your shirt!")
else:
    print("Hello friend! Why didn't you wear your red shirt today?")

# For loop on a list
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    print(x)

# While loop (Fibonacci series up to n)
def fib(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()
fib(1000)

"""

result=splitter.split_text(text)
print(result[0])
