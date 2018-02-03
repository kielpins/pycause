import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import StringIO
import pymc.graph

def pymc_plot_graph(myModel):
    
    gr = pymc.graph.graph(myModel) # myModel is a PyMC Model object
    png_str = gr.create_png(prog='dot')
    
    # treat the dot output string as an image file
    sio = StringIO.StringIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    
    # plot the image
    imgplot = plt.imshow(img, aspect='equal')
    plt.show(block=False)
