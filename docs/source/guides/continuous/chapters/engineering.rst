

Applications in Engineering
============================

Optimization is fundamental to mechanical engineering design. optymus provides tools for solving common engineering optimization problems.

Problem Types
-------------

**Structural Optimization**
   Minimize compliance (maximize stiffness) or weight subject to stress/displacement constraints.

**Topology Optimization**
   Determine optimal material distribution within a design domain. optymus includes PolyMesher for polygonal mesh generation.

**Shape Optimization**
   Optimize boundary geometry using signed distance functions.

**Size Optimization**
   Find optimal dimensions (thickness, cross-section) for structural members.

Pre-built Domains
-----------------

optymus includes classical benchmark domains with boundary conditions:

- **MBB Beam**: Simply supported beam with point load
- **Michell Truss**: Cantilever with circular support
- **Cook Membrane**: Tapered panel under shear
- **Wrench**: Mechanical component with applied torque
- **Suspension**: Automotive suspension component

Example usage:

.. code-block:: python

   from optymus.benchmark import MbbDomain
   from optymus.methods import polymesher

   result = polymesher(domain=MbbDomain, num_elements=100)

See Also
--------

- :doc:`/examples/index` - Mechanical engineering examples
- :doc:`/reference/docs_benchmark` - Domain API reference
