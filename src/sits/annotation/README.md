# sits.annotation

Modulo de anotacao manual para series temporais de satelite.

## Proposito

Sistema para anotacao manual de pixels, com diferentes estrategias de amostragem e persistencia automatica.

## Arquivos

| Arquivo | Descricao |
|---------|-----------|
| `store.py` | Armazenamento e persistencia de anotacoes |
| `samplers.py` | Estrategias de amostragem (random, grid, stratified, cluster) |
| `manager.py` | Gerenciador de sessao de anotacao |

## Uso Rapido

```python
from sits.annotation import AnnotationManager

# Criar sessao de anotacao
manager = AnnotationManager(
    session_dir="annotation/sessao_1",
    image_path="imagem_48bandas.tif",
    classification_path="classificacao.tif",
    target_class=1,  # Filtrar para classe 1
)

# Definir classes
manager.set_classes({
    "soja": 0,
    "milho": 1,
    "algodao": 2,
    "pasto": 3,
})

# Configurar amostragem
manager.set_sampler("random", seed=42)

# Navegar e anotar
coords = manager.go_to_next()
ndvi = manager.get_ndvi_series()
# ... visualizar ndvi ...
manager.annotate("soja")

# Estatisticas
print(manager.get_annotation_summary())

# Exportar dataset
manager.export_dataset()
```

## Classes

### AnnotationStore

Armazena e persiste anotacoes automaticamente.

```python
from sits.annotation import AnnotationStore, AnnotationResult

store = AnnotationStore(save_path="annotations.json", autosave=True)

# Adicionar anotacao
store.add(row=100, col=200, class_name="soja", class_id=0)

# Pular pixel
store.add(row=101, col=201, result=AnnotationResult.SKIPPED)

# Estatisticas
stats = store.get_statistics()
print(f"Total: {stats['total']}")
print(f"Por classe: {stats['by_class']}")

# Exportar
store.export_dataset(image_data, "samples.npz")
```

### Samplers

Diferentes estrategias de navegacao:

```python
from sits.annotation import RandomSampler, GridSampler, StratifiedSampler

# Aleatorio
sampler = RandomSampler(mask, seed=42)
coords = sampler.get_next()  # (row, col)

# Grid sistematico
sampler = GridSampler(mask, step=10)
coords = sampler.get_next()

# Estratificado por classe
sampler = StratifiedSampler(mask, classification, seed=42)
coords = sampler.get_next()  # Alterna entre classes

# Por cluster
sampler = ClusterSampler(mask, cluster_labels, seed=42)
coords = sampler.get_next()  # Alterna entre clusters
```

### AnnotationManager

Gerenciador completo de sessao:

```python
from sits.annotation import AnnotationManager

manager = AnnotationManager(
    session_dir="annotation/sessao_1",
    image_path="imagem.tif",
)

# Configurar
manager.set_classes({"soja": 0, "milho": 1})
manager.set_sampler("stratified")

# Navegar
manager.go_to_next()          # Proxima amostra
manager.go_to(row=100, col=200)  # Ir para coordenada

# Obter dados
pixel = manager.get_pixel_data()       # Todas as bandas
ndvi = manager.get_ndvi_series()       # Serie NDVI

# Anotar
manager.annotate("soja")      # Confirmar classe
manager.skip()                # Pular
manager.mark_uncertain()      # Marcar como incerto

# Desfazer
manager.undo_last()

# Exportar
manager.export_dataset("samples.npz")
```

## Exemplo Completo com Notebook

```python
import matplotlib.pyplot as plt
from sits.annotation import AnnotationManager

# Inicializar
manager = AnnotationManager(
    session_dir="annotation/sessao_1",
    image_path="imagem_48bandas.tif",
    classification_path="classificacao.tif",
    target_class=1,
)

manager.set_classes({
    "soja_1ciclo": 0,
    "soja_2ciclos": 1,
    "milho": 2,
    "outro": 3,
})

manager.set_sampler("random", seed=42)

# Funcao para visualizar e anotar
def show_and_annotate():
    coords = manager.go_to_next()
    if coords is None:
        print("Fim das amostras!")
        return

    row, col = coords
    ndvi = manager.get_ndvi_series()

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(ndvi, 'o-')
    plt.title(f"Pixel ({row}, {col})")
    plt.xlabel("Timestep")
    plt.ylabel("NDVI")
    plt.ylim(-0.2, 1.0)
    plt.grid(True)
    plt.show()

    # Input do usuario
    print("Classes: 0=soja_1ciclo, 1=soja_2ciclos, 2=milho, 3=outro")
    print("Comandos: s=skip, u=uncertain, q=quit")

    choice = input("Classe: ").strip()

    if choice == 's':
        manager.skip()
    elif choice == 'u':
        manager.mark_uncertain()
    elif choice == 'q':
        return False
    elif choice in ['0', '1', '2', '3']:
        class_names = ["soja_1ciclo", "soja_2ciclos", "milho", "outro"]
        manager.annotate(class_names[int(choice)])

    return True

# Loop de anotacao
while show_and_annotate():
    print(manager.get_annotation_summary())
    print("-" * 40)

# Exportar ao final
manager.export_dataset()
```

## Estrutura da Sessao

```
annotation/
├── sessao_1/
│   ├── annotations.json    # Anotacoes persistidas
│   ├── classes.json        # Definicao de classes
│   └── samples.npz         # Dataset exportado
```

## Formato do Dataset Exportado

```python
data = np.load("samples.npz")

data["X"]           # (n_samples, n_bands)
data["y"]           # (n_samples,)
data["rows"]        # (n_samples,)
data["cols"]        # (n_samples,)
data["class_names"] # Lista de nomes das classes
```

## Dependencias

- numpy >= 1.24.0
- rasterio >= 1.3.0
- loguru >= 0.7.0
