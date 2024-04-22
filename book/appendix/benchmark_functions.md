---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Benchmark functions

## Without constraints

### Rastrigin function

$$f(x) = An + \sum_{i=1}^{n}(x_{1}^{2} - A\cos(2\pi x_{i}))$$

where $A=10$.

```{code-cell}
from optymus.utils import plot_function, rastrigin_function

f = rastrigin_function()
plot_function(f, title='Rastrigin Function', min=-5.12, max=5.12)
``` 

### Ackley function

$$f(x,y) = -20exp[-0.2\sqrt{0.5(x^{2}+y^{2})}]-exp(0.5(cos(2\pi x)+cos(2\pi y)))+exp(1)+20$$


### Eggholder function

$$f(x,y) = -(y+47)sin(\sqrt{|y+x/2+47|})-xsin(\sqrt{|x-(y+47)|})$$


### Cross-in-tray function


$$f(x,y) = -0.0001(|\sin(x)\sin(y)\exp(|100-\sqrt{x^{2}+y^{2}}\pi|)|+1)^{0.1}$$

### Sphere function

$$f(x) = \sum_{i=1}^{n}x_{i}^{2}$$

### Rosenbrock function

$$f(x,y) = (1-x)^{2}+100(y-x^{2})^{2}$$

### Beale function

$$f(x,y) = (1.5-x+xy)^{2}+(2.25-x+xy^{2})^{2}+(2.625-x+xy^{3})^{2}$$


### Goldstein–Price function

$$ f(x,y) = [1+(x+y+1)^{2}(19-14x+3x^{2}-14y+6xy+3y^{2})][30+(2x-3y)^{2}(18-32x+12x^{2}+48y-36xy+27y^{2})]$$

### Booth function

$$f(x,y) = (x+2y-7)^{2}+(2x+y-5)^{2}$$

### Styblinski–Tang function

$$f(x) = \sum_{i=1}^{n}(x_{i}^{4}-16x_{i}^{2}+5x_{i})/2$$


### McCormick funtion

$$f(x,y) = \sin(x+y)+(x-y)^{2}-1.5x+2.5y+1$$

## With constraints