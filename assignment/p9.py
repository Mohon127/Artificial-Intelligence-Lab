import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# -----------------------------
# Utility: Load and preprocess image
# -----------------------------
def load_and_preprocess(img_path, target_size, preprocess_fn):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_fn(x)
    return x

# -----------------------------
# Utility: Visualize feature maps
# -----------------------------
def visualize_feature_maps(model, layer_names, preprocessed_img, title_prefix):
    for layer_name in layer_names:
        intermediate_model = Model(inputs=model.input,
                                   outputs=model.get_layer(layer_name).output)
        feature_maps = intermediate_model.predict(preprocessed_img)

        # Plot first 16 feature maps
        square = 4
        fig, axes = plt.subplots(square, square, figsize=(8, 8))
        fig.suptitle(f"{title_prefix} - {layer_name}", fontsize=14)
        for i in range(square * square):
            ax = axes[i // square, i % square]
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

# -----------------------------
# Main Execution
# -----------------------------
def main():
    img_path = "/home/mohon/4_1/lab/ai_lab/test_img/hibiscus_redmi_50MP_1_1.jpg"  # Replace with your favorite image path

    # VGG16
    vgg_model = VGG16(weights='imagenet', include_top=False)
    vgg_img = load_and_preprocess(img_path, (224, 224), vgg_preprocess)
    vgg_layers = ['block1_conv1', 'block3_conv3', 'block5_conv3']
    visualize_feature_maps(vgg_model, vgg_layers, vgg_img, "VGG16")

    # ResNet50
    resnet_model = ResNet50(weights='imagenet', include_top=False)
    resnet_img = load_and_preprocess(img_path, (224, 224), resnet_preprocess)
    resnet_layers = ['conv1_conv', 'conv3_block4_out', 'conv5_block3_out']
    visualize_feature_maps(resnet_model, resnet_layers, resnet_img, "ResNet50")

    # InceptionV3
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    inception_img = load_and_preprocess(img_path, (299, 299), inception_preprocess)
    inception_layers = ['conv2d_1', 'mixed3', 'mixed7']
    visualize_feature_maps(inception_model, inception_layers, inception_img, "InceptionV3")

if __name__ == "__main__":
    main()
