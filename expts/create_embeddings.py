import sys
import os
from mendeleev import element
from pymatgen.core.periodic_table import Element
import torch
from ocpmodels.datasets.elemental_embeddings import EMBEDDINGS

blockmap = {'s': 1, 'p': 2, 'd': 3, 'f': 4}
embedding = [torch.zeros(9)]
missing_values = []
for an in range(1, 101):
    elem = element(an)
    symbol = elem.symbol
    gn = elem.group_id
    pn = elem.period
    en = elem.electronegativity(scale='sanderson')
    cr = elem.covalent_radius_cordero
    v = elem.nvalence()
    ion = elem.ionenergies[1]
    ea = elem.electron_affinity
    block = blockmap[elem.block]
    av = elem.atomic_volume

    if ea is None:
        ea = -3
    if gn is None:
        gn = Element(symbol).group
    if av is None:
        av = Element(symbol).molar_volume
    emb = [gn, pn, en, cr, v, ion, ea, block, av]
    if None in emb:
        emb = [0]*9
        missing_values.append(an)
    embedding.append(torch.tensor(emb).to(torch.float32))

print(missing_values)

torch.save(embedding, "cgcnn_embeddings.pt")
torch.save(torch.tensor(missing_values), "missing_embeddings.pt")
