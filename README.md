# This code aims to generate convex clusters 

 Python convex_clusters

---

Those called convex clusters looks like the structures below. Obviously, there will be a set of convex cluster containing 61 atoms, but how to get those structures? This code is aiming to solve the problem.

![This is a convex cluster containing 61 atoms][1]
  [1]: ./61_1.png
  
  ---
  Exmaple:
```python
import main_fun
import pickle
import numpy as np
max_atom = 15
# generate the convex cluster whose atoms are less than max_atom (including equavalent)
lists = main_fun.all_convex_below(max_atom)
output = open('data.pkl', 'wb')
pickle.dump(lists, output)
output.close()
print('The result has been saved in' + 'data.pkl')
# main_fun.draw_fig(lists, 'vaca_0_', len(lists))
# main_fun.write_poscar(lists, '_vaca_0_', len(lists))
n_start = 2
n_end = max_atom
max_vaca = 2
vacancy_list = main_fun.vacan(lists, max_vaca, n_end, n_start)
main_fun.draw_fig(vacancy_list, 'vaca_1_', len(vacancy_list))
main_fun.write_poscar(vacancy_list, '_vaca_1_', len(vacancy_list))
``` 
---
Here, using `all_convex_below` method in `main_fun` class will help you generate the convex cluster whose atoms are less than max_atom (including equavalent). using `draw_fig` method in `main_fun` will help draw theses structure.
