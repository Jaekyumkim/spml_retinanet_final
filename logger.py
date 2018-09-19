# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import pdb

import tensorflow as tf
import numpy as np
import scipy.misc 
from PIL import Image, ImageDraw, ImageFont
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_drawbox(self, images, boxes, labels, score):
        """Draw boxes on images by Yeacheol Kim

        Args:
            images: [#width, #height]
            boxes:  [#boxes, 4]
            labels: [#boxes]

        Return:
            box drawd images: [#width, #height]
        """
        num_boxes = len(boxes)
        if num_boxes > 10:
            num_boxes = 10
        for i in range(num_boxes):
            img_temp = scipy.misc.toimage(images[i])
            draw = ImageDraw.Draw(img_temp)
            for box_idx in range(len(boxes[i])):
                draw.rectangle(boxes[i][box_idx].tolist(), outline=(255,0,0))
                draw.text((int(boxes[i][box_idx][0]),int(boxes[i][box_idx][1])),\
                        str(float(score[i][box_idx])), (255,255,255))
#            pdb.set_trace()
#            img_temp.save('tesorboard_image_test.png')
            img_temp = scipy.misc.fromimage(img_temp)
            img_temp = np.transpose(img_temp, (2,0,1))
            images[i] = img_temp
        return images

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
