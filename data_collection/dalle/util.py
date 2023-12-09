from openai import OpenAI

def generate(prompt_input):
    '''
    1. Takes i nan prompt and return the images generated to an specified folder
    2. Returns file name
    '''
    #Create DALLE Object
    client = OpenAI()

    modiefied_prompt = "realistic " + prompt_input + " with white background"

    #Call for generation using DALLE
    response = client.images.generate(
        prompt=modiefied_prompt,
        n=1,
        size="256x256")
    

    with open(f"dolle_synthesis.txt", mode="a") as file:
            file.writelines(modiefied_prompt + "\n" + response.data[0].url + "\n" + "\n")
