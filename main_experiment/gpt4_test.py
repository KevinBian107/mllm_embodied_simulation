from openai import OpenAI

client = OpenAI()


# Shirt, basic q

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
      
    {"role": "system",
     "content": """In this task, you will read short passages and look at an image of an object.
Please rate how sensible it would be to take action described in the last sentence using the object in
the image in the context of the whole passage. The scale goes from 1 (virtual nonsense) to 7 (completely sensible).
Be sure to read the sentences carefully. Please respond only with a number between 1 and 7.
"""},

    {
      "role": "user",
      "content": [
        {"type": "text", "text": "After wading barefoot in the lake, Erik needed something to get dry.	He used the object in the image to dry his feet."},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://urldefense.com/v3/__https://upload.wikimedia.org/wikipedia/commons/0/01/Charvet_shirt.jpg__;!!Mih3wA!CY-kiW3Q3BzJ5fboiiSABeHMpkviKw_h-kWHojuC_O7iV47FP_3YPrbk2GBlqaqzqvrIkxJh4i62EEzkNHI$ ",
          },
        },
      ],
    }
  ],
  max_tokens=300
)

message_content = response.choices[0].message.content

print(message_content)
