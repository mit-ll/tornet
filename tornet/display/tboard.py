"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import io
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tornet.display import plot_radar


def log_image(data, score, filename, vars_to_plot, file_writer, step,):
    # Prepare the plot
    figure = create_image(data, score, filename, vars_to_plot)
    # Convert to image and log
    with file_writer.as_default():
        tf.summary.image(filename, plot_to_image(figure), step=step)

        
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png',dpi=100)
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def create_image(data, score, filename, vars_to_plot):
    """
    Creates radar visualization
    """
    
    fig = plt.figure(figsize=(12,6))

    plot_radar(data,
                fig=fig,
                channels=vars_to_plot,
                include_cbar=True,
                time_idx=-1, # show last frame
                n_rows=2, n_cols=3)
    fname=os.path.basename(filename)
    fig.text(.5, .05,  '%s, score=%f' % (fname,score), ha='center')


    return fig