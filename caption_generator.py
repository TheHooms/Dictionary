import os
import openai
from PIL import Image
import boto3

imagepath = '/Users/houman/_code/Inq1/img/fire.png'

if(os.path.isfile(imagepath)):
    print('File exists')
else:
    print('File does not exist')

aws_access_key_id=''
aws_secret_access_key=''

with open('key/openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key

with open('key/aws_access_key_id.txt', 'r') as accesskeyfile:
    aws_access_key_id = accesskeyfile.read()

with open('key/aws_secret_access_key.txt', 'r') as secretfile:
    aws_secret_access_key = secretfile.read()

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name='us-west-1'
)
client = session.client('rekognition')

def create_alt_text(image_path):
    
    if not os.path.exists('_temp'):
        os.makedirs('_temp')
    
    img = Image.open(image_path)  

    new_size = (150, 150)  

    resized_img = img.resize(new_size)

    resized_img.save('_temp/temp.png') 
    
    with open('_temp/temp.png', 'rb') as image:
        response = client.detect_labels(Image={'Bytes': image.read()})
    
    image_labels = []
    for _label in response['Labels']:
            print( 'Label: ' , _label['Name'] , 'Confidence: ', _label['Confidence'] )
            if _label['Confidence']>75:
                image_labels.append(_label['Name'].lower())

    prompt = 'Provide two alternative captions, suitable for alt-text, based on the given image labels. Each caption should be under 10 words and should avoid referencing gender or age.: ' + ', '.join(image_labels)

    response = openai.Completion.create(
      model='text-davinci-003',
      prompt=prompt,
      temperature=0.8,
      max_tokens=128
    )

    captions = response['choices'][0]['text']
    
    output = 'Captions:\n'+captions

    return output

result = create_alt_text(imagepath)

print (result)
