from optymus.methods.topological import polymesher
from optymus.benchmark import HornDomain, MichellDomain

node, element, bs, bl, init_points = polymesher(domain=MichellDomain, n_elements=100, max_iter=1000, anim=True)