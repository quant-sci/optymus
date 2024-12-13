from optymus.methods import polymesher
from optymus.benchmark import HornDomain, MichellDomain

res = polymesher(domain=HornDomain, n_elements=100, max_iter=100, anim=True)