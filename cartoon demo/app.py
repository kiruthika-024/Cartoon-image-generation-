import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import os

torch.set_grad_enabled(False)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load(
        "bryandlee/animegan2-pytorch",
        "generator",
        pretrained="face_paint_512_v2",
        trust_repo=True
    ).to(device).eval()

    return model, device


model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def cartoonize_image(image):
    image.thumbnail((2048, 2048))
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = output_tensor.squeeze().cpu()
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)

    output_image = transforms.ToPILImage()(output_tensor)
    output_image = output_image.resize((1024, 1024), Image.BICUBIC)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output_image


st.title("🎨 Cartoon Yourself App")
st.write("Upload a photo and turn it into a cartoon using AnimeGANv2.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original Image")
    st.image(input_image, use_container_width=True)

    if st.button("✨ Cartoonize"):
        with st.spinner("Cartoonizing your image..."):
            cartoon_image = cartoonize_image(input_image)

        st.subheader("Cartoon Image")
        st.image(cartoon_image, use_container_width=True)

        os.makedirs("output", exist_ok=True)
        output_path = "output/cartoon.png"
        cartoon_image.save(output_path)

        st.success("✅ Cartoon image created successfully!")

        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Download Cartoon Image",
                f,
                file_name="cartoon.png",
                mime="image/png"
            )
