from __future__ import print_function, division
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import cv2
import telepot
from telepot.loop import MessageLoop
from openai import OpenAI


global ASSISTANT_ID
ASSISTANT_ID = "asst_Sa4rNiUTomxX1jUtUfk0YrGx"

global client
client = OpenAI(api_key="sk-proj-xqgMmLnCeFNqg9otBcmJT3BlbkFJfa3dH7sANoR4dQULko5u")


global device
device="cpu"

global model
model=torch.load('bobred.pth', map_location=torch.device('cpu'))

model.to(device)
model.eval()

global classes
classes= ('FAUCET', 'OUTLET', 'WALL')

global context
context = "Esta es la conversación con el usuario que llevas hasta ahora:"

global ultimoUser
ultimoUser = None

global loader
loader = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def image_loader(image_name):
    global device
    image = Image.fromarray(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image.to(device)

def handle(msg):
    global model
    global loader
    global classes
    global context
    global ultimoUser

    

    if 'text' in msg:
        if msg['from']['id'] != ultimoUser:
            context = "Esta es la conversación con el usuario que llevas hasta ahora:"
            context = context + "\n" + msg['from']['first_name'] + ": " + msg['text']
            ultimoUser = msg['from']['id']
        else:
            context = context + "\n" + msg['from']['first_name'] + ": " + msg['text']
        
        command=msg['text']

        if command=='Hola' or command=='hola' or command=='Hola!' or command=='/start':
            bot.sendMessage(msg['from']['id'], 'Hola '+ msg['from']['first_name'] +' soy Bob, tu asistente virtual')
            bot.sendMessage(msg['from']['id'], '¿En que puedo ayudarte?')
        else:
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "assistant",
                        "content": context,
                    },
                    {   
                        "role": "user",
                        "content": f"{command}",
                    }
                ]
            )
            run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)
            while run.status != "completed":
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                time.sleep(0.2)
            message_response = client.beta.threads.messages.list(thread_id=thread.id)
            messages = message_response.data
            latest_message = messages[0]
            bot.sendMessage(msg['from']['id'], latest_message.content[0].text.value)
    else:
        bot.sendMessage(msg['from']['id'], 'Foto recibida, analizaré tu problema')
        bot.sendMessage(msg['from']['id'], 'Estoy procesando la imagen, por favor espera')
        command=msg['photo'][2]['file_id']
        bot.download_file(command, 'fotoRecibida.png')
        
        img=cv2.imread('fotoRecibida.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image=image_loader(img)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
      
        print(classes[predicted[0]])

        if classes[predicted[0]] == 'FAUCET':
            bot.sendMessage(msg['from']['id'], '!Ash! Ya veo el problema con tu llave, puede que sea algo interno')
            bot.sendMessage(msg['from']['id'], 'No te preocupes, con BOB puedes encontrar un especialista en plomería en tu área')
            bot.sendMessage(msg['from']['id'], 'Por favor, ve al siguiente enlace para encontrar un especialista en plomería : ')
            bot.sendMessage(msg['from']['id'], 'https://www.plomeros.com')
        elif classes[predicted[0]] == 'OUTLET':
            bot.sendMessage(msg['from']['id'], '!Ash! Ya veo el problema con tu enchufe, cuidado con la electricidad')
            bot.sendMessage(msg['from']['id'], 'No te preocupes, con BOB puedes encontrar un especialista en electricidad en tu área')
            bot.sendMessage(msg['from']['id'], 'Por favor, ve al siguiente enlace para encontrar un especialista en electricidad : ')
            bot.sendMessage(msg['from']['id'], 'https://www.electricistas.com')
        elif classes[predicted[0]] == 'WALL':
            bot.sendMessage(msg['from']['id'], '!Ash! Ya veo el problema con tu pared, puede que sea algo estructural')
            bot.sendMessage(msg['from']['id'], 'No te preocupes, con BOB puedes encontrar un especialista en construcción en tu área')
            bot.sendMessage(msg['from']['id'], 'Por favor, ve al siguiente enlace para encontrar un especialista en construcción : ')
            bot.sendMessage(msg['from']['id'], 'https://www.constructores.com')
        
        bot.sendMessage(msg['from']['id'], 'Cuéntame, ¿En que más puedo ayudarte?')

        
bot = telepot.Bot('6814032942:AAHAB3RGrI5T6zMfPXE9C40Ehmhh_dhj6NI') #ChatBOB
MessageLoop(bot,handle).run_forever()






