"""
Busca TODOS los nombres que app.py importa (o usa) de la carpeta modules/ y que no existen.
Así no tienes que arreglar un ImportError a la vez.

Uso (desde la carpeta del proyecto, donde está app.py):
    python3 verificar_imports.py            # revisa app.py
    python3 verificar_imports.py otro.py    # revisa otro archivo
"""
import ast
import importlib
import os
import sys

ruta = sys.argv[1] if len(sys.argv) > 1 else "app.py"
carpeta = os.path.dirname(os.path.abspath(ruta))
sys.path.insert(0, carpeta)

with open(ruta, encoding="utf-8") as f:
    arbol = ast.parse(f.read())

problemas = []
importados = {}          # nombre usado en app.py -> objeto real (si se pudo importar)
revisados = 0

for nodo in ast.walk(arbol):
    if isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.module.split(".")[0] == "modules":
        try:
            mod = importlib.import_module(nodo.module)
        except Exception as e:                                   # noqa: BLE001
            problemas.append(f"El módulo '{nodo.module}' no se pudo importar: {type(e).__name__}: {e}")
            continue
        for alias in nodo.names:
            revisados += 1
            if hasattr(mod, alias.name):
                importados[alias.asname or alias.name] = getattr(mod, alias.name)
            else:
                problemas.append(f"'{alias.name}' NO existe en {nodo.module}   (archivo: {getattr(mod, '__file__', '?')})")

# También revisa los métodos de clases importadas, por ejemplo SystemAnalyzer.analyze_preset
for nodo in ast.walk(arbol):
    if isinstance(nodo, ast.Attribute) and isinstance(nodo.value, ast.Name) and nodo.value.id in importados:
        objeto = importados[nodo.value.id]
        if isinstance(objeto, type) and not hasattr(objeto, nodo.attr):
            revisados += 1
            problemas.append(f"El método '{nodo.value.id}.{nodo.attr}' NO existe (lo usa app.py en la línea {nodo.lineno})")

print(f"Revisé {revisados} nombres en {ruta}.")
if problemas:
    print(f"\nProblemas encontrados ({len(problemas)}):")
    for p in dict.fromkeys(problemas):
        print("  -", p)
    sys.exit(1)
print("Todo lo que app.py importa de modules/ existe.")
