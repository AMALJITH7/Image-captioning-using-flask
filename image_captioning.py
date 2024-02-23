from flask import Flask
from flask import render_template,request


from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import AutoProcessor, AutoModelForObjectDetection
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import base64



app = Flask(__name__)


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams }

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        image_path = image_path.strip()
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)  # Changed skip_special_tokenizer to skip_special_tokens
    preds = [pred.strip() for pred in preds]

    # Remove square brackets and single quotes
    preds = [pred.strip("[]' ") for pred in preds]

    return preds


@app.route('/')
def start():
      return render_template('front.html')


@app.route('/click', methods=['POST'])
def process():
      img= request.form['input']
      out = str(predict_step([img]))
      result = out.strip("[]' ")
      image = cv2.imread(img)
      pic = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      plt.imshow(pic)
      plt.title(f'Predicted caption according to the picture : {result}')
      plt.axis('off')
      plt.gcf().set_size_inches(15, 10)
      plt.show()
      return render_template('front.html')
     

if __name__ == '__main__':
    app.run(debug=True)
