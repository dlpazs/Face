# CSCI 512

[Link](http://inside.mines.edu/~whoff/courses/EENG512/)
[Link](https://www.youtube.com/watch?v=skaQfPQFSyY&list=PL7v9EfkjLswLfjcI-qia-Z-e3ntl9l6vp&index=2&t=0s)

- TBC


[Link](https://www.youtube.com/watch?v=2S4nn7S8Hk4&list=PLAwxTw4SYaPnbDacyrK_kB_RUkuxQBlCm)

- Difference between Computer Vision and Computational Photography. Image processing is manipulating images.
- What is computer vision? Interpret images. Understand something about an image.
- Why is this hard? Seeing is not the same as measuring intensity. Seeing is built perception based upon measurements
made by an imaging sensor.
- perception is an active construction. Your brain makes the description of that construction.
- Triangle of CV: Computational model (math), algorithm, real images (scene/ground truth).


## Images as Functions : 

- image is I(x, y) as a function of intensity. Intense bright parts are in the foreground and dark in a surface. If you smooth the function the surface will be more smoove! 
- images go from some min to max (0-255 is a mistake and caused by byte length)
- An image is a collection of intesities and can be described as a function, and can be described by X and Y. 
- So an image is a function of the X(min, max) x Y(min, max) -> image intensities range (0,10) as an example 
- A colour image is also a function, instead of a range it is a vector of different channels it is a mapping of R2 to R3
- in a computer it has to be digital. 
- digital images are discrete images: sample 2D space on regular grid, then quantize each sample round to nearest integer
- Have to be floats uint breaks
- row i column j, x horizontal y vertical 
- since an image is sampled at discrete locations in space, it can be written as a 2D array or matrix. 
- images are just two functions thus we can just add them
- noise in an image is just another function
