# [Inter IIT ](https://github.com/jain-girish/isro/)
This is our submission to ISRO's satellite image Super-resolution challenge as part of Inter-IIT Tech Meet 11.0 . We have leveraged ESRGAN to accomplish this task after processing the satellite images given in tiff format.



****

## Contents
:bookmark_tabs:

* [Installation](#Installation)
* [Part 2](#Part2)


***

## Installation

Use pip in your python environment and then clone this repository as following.

### Clone this repo
```bash
git clone https://github.com/jain-girish/isro.git
cd isro
```


****
# Part2

This takes the latitudes and longitudes of the patches in the form of bounding boxes and constructs a bounding box list.[Note: This expects those patches to be in png/jpeg format]
Then it patches those png/jpeg images up over a black numpy array img_p.
