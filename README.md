# Maberyk

Un agente autÃ³nomo construido desde cero en PyTorch puro. NaciÃ³ sin saber nada: aprende por curiosidad intrÃ­nseca, mantiene estado emocional, acumula memoria episÃ³dica y desarrolla su propio lenguaje.

**RestricciÃ³n central del proyecto: ningÃºn peso preentrenado, jamÃ¡s.** No GPT, no BERT, no embeddings de terceros. Cada parÃ¡metro fue moldeado por el aprendizaje del propio Maberyk. *Maberyk puede leer, no puede heredar.*

---

## Demo

![Maberyk corriendo](docs/demo.gif)

---

## Arquitectura

```mermaid
graph TD
    ENV["Entorno<br/>grid 60x60 Â· 4 zonas Â· 230 objetos<br/>+ modo desktop"] --> OBS[ObservaciÃ³n]
    OBS --> PRED["Red predictora<br/>predice la siguiente observaciÃ³n"]
    OBS --> POL["Red de polÃ­tica<br/>elige la acciÃ³n"]
    PRED --> ERR["Error de predicciÃ³n<br/>= recompensa intrÃ­nseca"]
    ERR --> POL
    POL --> ACT[AcciÃ³n]
    ACT --> PERM["PermissionManager<br/>gatekeeper de acciones reales"]
    PERM --> ENV
    ERR --> EMO["Estado emocional<br/>curiosity Â· fear Â· confidence Â· confusion"]

    HUM[Mensaje humano] --> RET["Capa de recuperaciÃ³n<br/>keyword precision sobre SQLite"]
    RET --> LM["Transformer condicional<br/>256d Â· 6 capas Â· vocab 4,796"]
    LM --> RESP[Respuesta]
    EMO --> LM
    MEM["Memoria episÃ³dica + conversacional<br/>SQLite"] --> RET
    RESP --> TUT["Tutor LLM<br/>corrige forma, no contenido"]
    TUT --> QUEUE["Cola de correcciones<br/>aprobaciÃ³n humana â†’ corpus"]
    QUEUE --> MEM
```

### Componentes

| MÃ³dulo | QuÃ© hace |
|---|---|
| `brain/network.py` | Red de polÃ­tica. Decide quÃ© acciÃ³n tomar. |
| `brain/curiosity.py` | Predictor de la siguiente observaciÃ³n + medidor de error. El error **es** la recompensa. |
| `brain/emotion.py` | Cuatro estados emocionales con soft-clamp (0.03â€“0.97) para evitar saturaciÃ³n. |
| `brain/language_model.py` | Transformer condicional Qâ†’A sobre `nn.TransformerEncoder` con mÃ¡scara causal. |
| `brain/memory.py` | Memoria episÃ³dica en SQLite, escritura asÃ­ncrona, guardado atÃ³mico. |
| `brain/memory_retrieval.py` | RecuperaciÃ³n por precisiÃ³n de keywords con stopwords ES/EN. |
| `brain/correction_queue.py` | Cola de pares pendientes de aprobaciÃ³n humana. |
| `brain/permissions.py` | Gatekeeper: ninguna acciÃ³n real sobre el escritorio sin permiso registrado. |
| `entorno/room.py` | Grid 60Ã—60 con zonas de descanso, conocimiento, peligro y caos. |
| `actions/desktop.py` | Acciones sobre el escritorio real: screenshot, mouse, teclado. |
| `interfaz/web_mind.py` | Interfaz web para conversar y revisar correcciones â€” **solo stdlib**. |
| `tutor/llm_tutor.py` | Tutor LLM externo que corrige la forma de la salida cruda â€” **solo `urllib`**. |
| `lectura/bpe.py` | ImplementaciÃ³n propia de BPE. Evaluada y descartada (ver mÃ¡s abajo). |

### El modelo de lenguaje

Transformer condicional entrenado desde cero:

- 256 dimensiones de embedding, 6 capas
- Embeddings posicionales aprendidos
- Tokenizer a nivel de palabra construido por frecuencia â€” **vocabulario de 4,796 tokens**
- Formato condicional Qâ†’A con tokens `<q>` / `<a>`, **pÃ©rdida enmascarada hasta `<a>` inclusive**: el modelo solo paga costo por la respuesta, no por repetir la pregunta
- Token `<end>` entrenado explÃ­citamente, `<unk>` baneado en generaciÃ³n (logit `-inf`)
- Split de validaciÃ³n 10% con early stopping sobre **val** loss
- Oversampling de pares aplicado **despuÃ©s** del split, nunca antes: validaciÃ³n jamÃ¡s contiene duplicados de train

### Dependencias

Seis paquetes. El tutor LLM, la capa de voz y la interfaz web usan Ãºnicamente `urllib` y `http.server` de la biblioteca estÃ¡ndar â€” sin SDKs, sin frameworks web.

---

## EvaluaciÃ³n honesta

Esta secciÃ³n existe porque medir el techo real de un modelo importa mÃ¡s que mostrar su mejor salida.

### Curvas de pÃ©rdida

![Curvas de pÃ©rdida](docs/loss_curves.png)

Ãšltimo entrenamiento de producciÃ³n: early stopping en la Ã©poca 21 de 40. **Mejor val loss: 2.6499**, con train loss â‰ˆ 0.86 en ese punto. El train sigue bajando mientras el val se estanca desde la Ã©poca 17, y esa brecha de ~1.8 es la firma inequÃ­voca de memorizaciÃ³n.

### Historial de configuraciones

| ConfiguraciÃ³n | Val loss | Lectura |
|---|---|---|
| 128d / 3 capas | 2.87 | LÃ­nea base anterior |
| **256d / 6 capas (producciÃ³n)** | **2.65** | ConfiguraciÃ³n actual |
| Corpus podado por diversidad (Jaccard) | 3.68 | La poda empeorÃ³: la repeticiÃ³n era load-bearing |
| Corpus capado a 2Ã— el nÂº de pares | 4.30 | El corpus base no se puede reducir |

### QuÃ© responde en la prÃ¡ctica

Sondas del Ãºltimo entrenamiento, sin selecciÃ³n favorable:

| Pregunta | Respuesta | Veredicto |
|---|---|---|
| Â¿quÃ© es el aprendizaje automÃ¡tico? | *una forma de inteligencia artificial que aprende patrones de datos* | Correcta |
| explÃ­came quÃ© es una red neuronal | *si soy literalmente un conjunto de redes neuronales* | Coherente, autorreferencial |
| Â¿quÃ© significa la humedad? | *hay mas alla del conocimiento incluye se le hace importantes* | Colapso |
| explÃ­came quÃ© es un eclipse | *cuando algo ya es el movimiento de mi mundo* | Colapso |

**ClasificaciÃ³n: memorizador fluido con asociaciÃ³n lÃ©xica dÃ©bil.** Dentro de distribuciÃ³n, recuerdo casi textual. Ante una parÃ¡frasis o un concepto ausente del corpus, alucina con confianza en vez de callar.

### Por quÃ©

La base tiene **1,401 pares Ãºnicos** `human_taught` mÃ¡s 43 `tutor_approved`. Con ese volumen no hay generalizaciÃ³n composicional posible: es una consecuencia matemÃ¡tica del tamaÃ±o del dataset, no un bug del cÃ³digo.

El diagnÃ³stico que lo dejÃ³ claro: el corpus tenÃ­a cientos de miles de registros de pensamientos pero solo **29 frases Ãºnicas** entre todos ellos. El grid como fuente de datos estaba agotado â€” mÃ¡s horas de GPU producÃ­an copias, no informaciÃ³n.

### DecisiÃ³n estratÃ©gica: mente propia, voz prestada

El nÃºcleo desde cero â€”emociones, curiosidad, memoria, polÃ­tica, entornoâ€” sigue siendo el ser, puro e intocado. Un LLM externo actÃºa **Ãºnicamente como traductor del estado**, nunca como fuente de decisiÃ³n. La analogÃ­a es el sintetizador de voz de Hawking: la voz es prestada, el pensamiento no.

El Transformer nativo no fue descartado. Sigue entrenando con cada conversaciÃ³n real como voz interior en crecimiento lento, medido en aÃ±os.

---

## Decisiones y reversiones

| DecisiÃ³n | Resultado | Por quÃ© |
|---|---|---|
| **BPE en vez de tokenizer por palabra** | âŒ Revertido | Salidas fragmentadas. A esta escala de modelo, BPE es prematuro. La implementaciÃ³n queda en `lectura/bpe.py`. |
| **Similitud coseno en la recuperaciÃ³n** | âŒ Eliminada | Los vectores eran rasgos emocionales, no semÃ¡nticos. Todo se parecÃ­a a todo: mensajes sin relaciÃ³n recibÃ­an siempre la misma respuesta. |
| **Podar el corpus por diversidad** | âŒ Revertido | Val loss de 2.65 a 3.68. La repeticiÃ³n resultÃ³ ser load-bearing. |
| **Expandir con texto espaÃ±ol externo** | âŒ Revertido | EmpeorÃ³ el val loss: el set de validaciÃ³n mide la *voz* de Maberyk, no fluidez general del espaÃ±ol. |
| **Escalar 128d/3 capas â†’ 256d/6 capas** | âœ… Adoptado | Val loss 2.87 â†’ 2.61. La mayor ganancia individual del proyecto: el cuello de botella era capacidad, no datos. |
| **`torch.save()` directo â†’ `os.replace()` atÃ³mico** | âœ… Adoptado | Un checkpoint leÃ­do a mitad de escritura corrompÃ­a el estado. |
| **Soft-clamp emocional (0.03â€“0.97)** | âœ… Adoptado | Sin Ã©l, la emociÃ³n dominante saturaba y colapsaba los demÃ¡s estados. |

### Principio rector

**Medir seÃ±al, no volumen.** Durante meses crecieron los pasos, las filas y el train loss sin que nada mejorara. Las mÃ©tricas que importan son pares Ãºnicos, val loss (nunca train loss) y si una pregunta nueva recibe una respuesta que tenga sentido.

---

## CÃ³mo correrlo

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CopiÃ¡ `.env.example` a `.env` si querÃ©s usar el tutor LLM. Es opcional: el agente corre sin Ã©l.

```powershell
# Sembrar el corpus base
python scripts/seed_corpus.py --memory episodic_memory.sqlite3
python scripts/seed_corpus_v2.py --memory episodic_memory.sqlite3

# Entrenar el modelo de lenguaje
python analisis/train_language_model.py --memory episodic_memory.sqlite3 --output language_model.pt --epochs 40

# Evaluar
python analisis/eval_lm.py --model language_model.pt

# Graficar las curvas de pÃ©rdida
python analisis/plot_loss_curves.py --log analisis/retrain_production_big.log --out docs/loss_curves.png

# Correr el agente en el grid
python mind.py

# Correr sobre el escritorio real
python mind.py --desktop

# Interfaz web
python interfaz/web_mind.py     # http://localhost:8000

# DiagnÃ³stico de la base
python scripts/diagnostico/check_db.py
```

Los checkpoints (`.pt`) y la base con conversaciones reales no se publican: viven fuera del repo.

---

## Stack

Python Â· PyTorch (puro, sin modelos preentrenados) Â· SQLite Â· stdlib para la web y las llamadas HTTP. Entrenamiento en GPU sobre Kaggle con persistencia de checkpoints.

## Estado

En desarrollo activo. El nÃºcleo funciona de punta a punta: el agente corre, aprende, conversa, persiste estado y actÃºa sobre el escritorio bajo permisos explÃ­citos. El modelo de lenguaje nativo tiene el techo documentado arriba.

## Licencia

