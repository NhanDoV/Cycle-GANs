from summer_winter_cycle_gans import *
import logging, os, cv2
from flask import Flask, render_template, request

app = Flask(__name__)
@app.route('/')

def home():
    return render_template('home.html')

# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")

towinter_model =  CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)
tosummer_model =  CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_Y, discriminator_Y=disc_X
)

towinter_model.load_weights("translated_models/summer2winter/summer2winter").expect_partial()
tosummer_model.load_weights("translated_models/winter2summer/winter2summer").expect_partial()

print("Weights loaded successfully")

@app.route('/' , methods = ['GET', 'POST'])
def main():

    if request.method == 'POST':

        transform_type = request.form['transformed_type']
        img_file = request.files['file']
        img_name = os.path.join('static', 'input.png')
        img_file.save(os.path.join('static', 'input.png'))

        image = tf.io.read_file(img_name)
        image_c = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize([image_c], [256,256])
        w, h = image_c.shape[:2]

        if transform_type == 'to_summer':
           prediction = tosummer_model.gen_G(image, training=False)[0].numpy()
        else: 
           prediction = towinter_model.gen_G(image, training=False)[0].numpy()

        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        pred_img = cv2.resize(prediction.reshape(256,256,3), (h, w))
        output_fname = os.path.join('static', 'output.png')
        cv2.imwrite(output_fname, (pred_img*255).astype(np.uint8))

        return render_template('home.html', 
                                uploaded_image = img_name,
                                transformed_img = output_fname
                            )

if __name__ == "__main__":
    app.run(port='8080', threaded=False, debug=True)