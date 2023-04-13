Learn how to change the background of any photo using SAM (Segment Anything Model) + Stable-Diffusion!

SAM is the new open-source Meta AI designed for precise object segmentation 🤯.

I've adapted the code to perform two main tasks:
1. Segment and generate a mask for the chosen object.
2. Utilize Stable-Diffusion to create the new prompted background.

Access my GitHub repository to find the code: [GitHub link]

I ran this in T4 16vGPU instance in  #aws … and used Gradio for the UI.

Challenges faced:
* The mask isn't highly accurate, resulting in occasional noise.
* Stable-Diffusion sometimes alters the subject's composition, including facial features.

Potential Next Steps:
* Delve deeper into SAM to explore ways of achieving more accurate results.
* Experiment with alternative models, such as ControlNet, or explore other cutting-edge technologies, and today was released controlNet 1.1 🤯 so ...
* if get satisfactorily results, deploy it maybe in Replicate :)