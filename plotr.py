# Functions for plotting training losses using plotly
import plotly.graph_objects as go
import torchvision.transforms.functional as F
from PIL import Image
import os
from IPython.display import IFrame

# Basic loss plotting function
def plot(content_loss, style_loss, total_loss, filename='loss_plot.html'):
    # Create plotly figure
    fig = go.Figure()

    # Add each loss type as a separate line
    fig.add_trace(go.Scatter(x=list(range(len(content_loss))), y=content_loss, name='Content Loss'))
    fig.add_trace(go.Scatter(x=list(range(len(style_loss))), y=style_loss, name='Style Loss'))
    fig.add_trace(go.Scatter(x=list(range(len(total_loss))), y=total_loss, name='Total Loss'))

    # Add title and axis labels
    fig.update_layout(title='Loss Visualization during Training', xaxis_title='Iterations', yaxis_title='Loss')
    fig.write_html(filename)


# Plot losses averaged over batches to smooth the graph
def plot_batched_loss(content_losses, style_losses, total_losses, filename, batch_size=32):
    """
    Takes raw losses and plots them averaged over batches.
    Makes the trends easier to see by reducing noise.
    """
    # Calculate how many complete batches we have
    num_batches = len(content_losses) // batch_size

    # Lists to store averaged losses
    batched_content_losses = []
    batched_style_losses = []
    batched_total_losses = []

    # Average losses for each batch
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batched_content_losses.append(sum(content_losses[start_idx:end_idx]) / batch_size)
        batched_style_losses.append(sum(style_losses[start_idx:end_idx]) / batch_size)
        batched_total_losses.append(sum(total_losses[start_idx:end_idx]) / batch_size)

    # Create and save the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(num_batches)), y=batched_content_losses, name='Content Loss'))
    fig.add_trace(go.Scatter(x=list(range(num_batches)), y=batched_style_losses, name='Style Loss'))
    fig.add_trace(go.Scatter(x=list(range(num_batches)), y=batched_total_losses, name='Total Loss'))
    fig.update_layout(title='Loss Visualization during Training', xaxis_title='Batch Number', yaxis_title='Loss')
    fig.write_html(filename)

# Use these functions after training:
# plot(content_losses, style_losses, total_losses, 'my_plot.html')
# plot_batched_loss(content_losses, style_losses, total_losses, 'batched_plot.html')



def show_plot(filepath):
    """Displays a Plotly HTML file in a Jupyter Notebook using an IFrame."""
    return IFrame(filepath, width="100%", height="600")  # Adjust width and height as needed



def interpolate_image(input_path, output_path, target_size, interpolation_mode="bicubic"):
    """
    Interpolates an image to a specified size using PyTorch.

    Args:
        input_path: Path to the input image file.
        output_path: Path to save the interpolated image.
        target_size: A tuple (height, width) representing the desired size.
        interpolation_mode: Interpolation mode. Options: "nearest", "linear", "bilinear", "bicubic", "area", "nearest-exact". Defaults to "bicubic".
    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: if the interpolation mode is invalid.
        RuntimeError: If there is an issue during image processing.
    """
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        img = Image.open(input_path).convert("RGB")  # Ensure RGB for consistent tensor conversion
        img_tensor = F.to_tensor(img).unsqueeze(0)  # Add batch dimension

        interpolation_modes = {
            "nearest": F.InterpolationMode.NEAREST,
            # "linear": F.InterpolationMode.LINEAR,
            "bilinear": F.InterpolationMode.BILINEAR,
            "bicubic": F.InterpolationMode.BICUBIC,
            # "area": F.InterpolationMode.AREA,
            "nearest-exact": F.InterpolationMode.NEAREST_EXACT
        }

        if interpolation_mode not in interpolation_modes:
            raise ValueError(
                f"Invalid interpolation mode: {interpolation_mode}. Must be one of {list(interpolation_modes.keys())}")

        resized_tensor = F.resize(img_tensor, target_size, interpolation=interpolation_modes[interpolation_mode])

        resized_img = F.to_pil_image(resized_tensor.squeeze(0))  # Remove batch dimension

        resized_img.save(output_path)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"A runtime error occurred during image processing: {e}. Check if the image is valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

