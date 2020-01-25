import numpy as np
import matplotlib.pyplot as plt

def create_circle(r, x_offset=0, y_offset=0):
    """
    Plots a circle with radius r centered at the point (x_offset, y_offset)
    
    Args:
        r (double): Radius of the circle, normalized to the width and height of the plot
        x_offset (double): x-coordinate of the circle's center, normalized to the width of the plot
        y_offset (double): y-coordinate of the circle's center, normalized to the height of the plot
        
    Returns:
        None
    """
    x = np.linspace(-1,1,1000)
    y = np.linspace(-1,1,1000)
    
    [X,Y] = np.meshgrid(x,y)
    circle = ((X-x_offset)**2 + (Y-y_offset)**2) < r
    
    plt.figure()
    plt.imshow(circle, origin = 'lower')


def create_ellipse(a, b, x_offset=0, y_offset=0, orientation = 'horizontal'):
    """
    Plots an ellipse with semi-major axis a and semi-minor axis b, centered at the point (x_offset, y_offset)
    
    Args:
        a (double): Semi-major axis of the ellipse, normalized to the width of the plot
        b (double): Semi-minor axis of the ellipse, normalized to the height of the plot
        x_offset (double): x-coordinate of the circle's center, normalized to the width of the plot
        y_offset (double): y-coordinate of the circle's center, normalized to the height of the plot
        orientation (string): Defines whether the semi-major axis lies along the x-axis (horizontal) or the y-axis (vertical)
        
    Returns:
        None
    """
    
    #The semi-major axis must always be larger than the semi-minor axis
    #Gracefully handles case where user accidentally passes a and b in the wrong order
    if b > a:
        a, b = b, a
    
    x = np.linspace(-1,1,1000)
    y = np.linspace(-1,1,1000)
    
    if orientation == 'horizontal':
        #Semi-major axis is in x-direction, giving the impression of a "horizontal" ellipse
        rx = a
        ry = b
    if orientation == 'vertical':
        #Semi-major axis is in y-direction, giving the impression of a "vertical" ellipse
        rx = b
        ry = a
        
    [X,Y] = np.meshgrid(x,y)
    ellipse = (((X-x_offset)/rx)**2 + ((Y-y_offset)/ry)**2) < 1
    
    plt.figure()
    plt.imshow(ellipse, origin = 'lower')
    
    
def create_rectangle(width, height):
    """
    Creates a rectangle in the center of the plot.
    
    Args:
        width (double): Width of the rectangle, normalized to the width of the plot
        height (double): Height of the rectangle, normalized to the height of the plot
    """
    w = int(width*1000)
    h = int(height*1000)
    
    grid = np.zeros([1000,1000])
    
    x = np.size(grid,1)
    y = np.size(grid,0)
    
    grid[(y - h)//2:(y + h)//2, (x - w)//2:(x + w)//2] = 1
    
    plt.figure()
    plt.imshow(grid, origin = 'lower')

    
def create_formed_half_space(x, y, function, inverted=False):
    """
    Defines a boundary separating the plane into two sections along the y-axis, and fills only one side of the boundary.
    
    Args:
        x (ndarray): Numpy array of points along the x-axis
        y (ndarray): Numpy array of points along the y-axis
        function (function): A function object, which will operate on x to obtain a boundary
        inverted (bool): False by default, wherein all elements below the boundary are set to 1. If inverted is True, all elements above the boundary are set to 1.
    
    Returns:
        None
    """
    [X,Y] = np.meshgrid(x,y)
    
    f = function(x)
    half_space = f > Y
    
    if inverted:
        half_space = ~half_space
        
    plt.figure()
    plt.imshow(half_space, origin = 'lower')
    

def line(m, b):
    """
    Returns a lambda expression for the equation of a line, with slope m and intercept b.

    Args:
        m (double): The slope of the line
        b (double): The intercept of the line

    Returns:
        A lambda expression
    """
    return lambda x: m*x + b


if __name__ == '__main__':
    create_circle(0.6)
    create_circle(0.3, x_offset=0.2, y_offset=0.4)
    
    create_ellipse(0.7, 0.3)
    create_ellipse(0.9, 0.1, orientation='vertical')
    create_ellipse(0.5, 0.5, x_offset=-0.3)
    
    create_rectangle(0.4, 0.8)
    create_rectangle(1, 0.3)
    
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.linspace(-2, 2, 1000)
    
    create_formed_half_space(x, y, np.sin)
    
    x = np.linspace(0, 10, 1000)
    y = np.linspace(0, 10, 1000)
    
    m = -2
    b = 7
    function = line(m, b)
    
    create_formed_half_space(x, y, function, inverted = True)