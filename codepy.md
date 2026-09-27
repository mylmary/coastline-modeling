```import os
import zipfile

zip_path = "/kaggle/input/coastline-data/GMTED2010S10E030_075.zip"
extract_dir = "/kaggle/working/gmted"

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zip_file:
    zip_file.extractall(extract_dir)

print("Extraction complete.")

for root, dirs, files in os.walk(extract_dir):
    for file in files:
        print(os.path.join(root, file))
```


To print a message to the console in Python, you use the `print()` function. 

Here is a quick example of how it works:

```python
def greet_user(name):
    # This function prints a personalized greeting
    message = f"Hello, {name}!"
    print(message)

greet_user("Alice")
```

After the code block, you can just continue typing your explanation normally here.
