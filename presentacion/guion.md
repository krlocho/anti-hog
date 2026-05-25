# Guion de defensa — Anti Hog

> Tono: natural, como si lo contaras a un colega. No leas palabra por palabra — quédate con las ideas y los números clave (en **negrita**). Duración objetivo: 8-10 min.

---

## 1 · Portada

Buenas. Mi proyecto se llama **Anti Hog**. La idea, en una frase: coger un canal público de Telegram donde la gente avisa de controles policiales en Lanzarote, y a partir de todo ese texto desordenado, construir un sistema que aprenda los patrones y prediga dónde es más probable que haya un control.

Os voy a enseñar el camino completo, desde los mensajes en crudo hasta la aplicación final.

*(Pausa, pasa de diapo.)*

---

## 2 · El problema

¿De dónde sale esto? En Lanzarote hay un canal de Telegram, activo desde 2015, donde los vecinos van escribiendo cosas como "Guardia Civil en la rotonda de Playa Blanca". Mensajes cortísimos, en lenguaje de la calle, sin ninguna estructura.

Mi pregunta de partida fue: **¿se puede sacar algo útil de ahí?** ¿Hay patrones reales — por día de la semana, por hora, por zona — y se pueden aprender para predecir?

Un apunte importante: esto lo planteo como un **ejercicio académico de análisis de datos públicos**. No es una herramienta para saltarse controles; de hecho los controles de hace años ya no existen. Es un caso de uso para demostrar técnicas de IA sobre datos reales y sucios.

---

## 3 · Los datos

El dataset son casi **16.000 mensajes**, de 2015 hasta hoy.

Aquí me encontré con una sorpresa que al principio me preocupó: hay un **hueco de cinco años**. El canal estuvo muy activo hasta 2021, luego se murió, y ha resucitado este mes. Esto es importante porque si entrenas un modelo metiendo esos cinco años vacíos, se rompe. Así que el sistema **detecta ese gap solo** y se queda con el periodo que sí tiene datos.

Y el otro reto: los mensajes son muy cortos, **44 caracteres de media**. Eso luego complica tanto la extracción como la búsqueda, y lo tuve que tener en cuenta.

---

## 4 · Pipeline

Antes de entrar en detalle, esta es la foto completa. Son **seis fases**, y cada una es un script independiente que se puede ejecutar por separado.

Primero **bajo los datos** de Telegram. Luego **extraigo la información** con un modelo de lenguaje. Después **geolocalizo** los lugares. Con eso ya **entreno el modelo** predictivo. En paralelo monto un **RAG** para poder preguntarle al histórico. Y todo acaba en una **aplicación**.

Voy fase por fase.

---

## 5 · Scraping

La primera fase es bajar los mensajes. Uso **Telethon**, que es la API oficial de Telegram, así que descargo el histórico completo de forma legítima.

Y aquí metí algo de lo que estoy contento: un **proceso programado que se ejecuta cada 20 horas** automáticamente. O sea, el dataset no es estático — crece solo mientras el canal siga activo. De hecho, entre que empecé y terminé, ya había capturado mensajes nuevos él solo.

Cada descarga se guarda como copia de seguridad y se fusiona sin duplicados, así que es imposible perder datos.

---

## 6 · Extracción NLP

Esta es para mí la fase más interesante. Tengo texto libre y necesito datos estructurados.

Lo que hago es pasar cada mensaje por **gpt-4o-mini** pidiéndole que me devuelva un JSON: si es un control o no, la fecha, la hora, el lugar y el tipo. Aquí lo veis: de "Gc rotonda playa blanca control alcoholemia" saco un objeto limpio con todos los campos.

Lo lancé en paralelo y con caché, para no repetir llamadas ni gastar de más. De los 16.000 mensajes, identifiqué **8.444 controles reales** — el resto eran saludos, off-topic, mensajes del grupo...

---

## 7 · Geocoding

Ahora, "rotonda del Gallo Loco" no le dice nada a un mapa. Necesito coordenadas.

Para eso construí un **diccionario de lugares de Lanzarote** a mano — más de 75 sitios, rotondas, pueblos — y uso *fuzzy matching* para los que vienen mal escritos.

Una decisión de diseño: hice **dos niveles de zonas**. Uno de 6 zonas grandes, que uso para el documento y las estadísticas, y otro de **12 zonas** más finas, que es el que come el modelo. Conseguí geolocalizar el **72%** de los eventos.

---

## 8 · Modelo

Y llegamos al modelo. La forma en que planteo el problema: divido el espacio y el tiempo en **celdas** — una celda es "esta fecha, esta zona, esta franja de 3 horas" — y el modelo predice la probabilidad de que haya un control en cada celda.

Uso **XGBoost**, con features como el día de la semana, el mes, si es finde, si es festivo, la zona...

Dos cosas de las que quiero presumir un poco. Una: **validación temporal**, no validación cruzada normal. Entreno con el pasado y valido con el futuro. Si hiciera k-fold normal estaría haciendo trampa, porque mezclaría días futuros en el entrenamiento. Y dos: usé **Optuna** para buscar los hiperparámetros y **early stopping** para no sobreentrenar.

El resultado es un **AUC de 0.79**. Que significa que, si cojo una celda con control y otra sin control al azar, el modelo le da más probabilidad a la correcta el 79% de las veces. Para un evento tan raro y con doce zonas, es un resultado sólido.

---

## 9 · RAG

Además del predictor, quería poder **preguntarle al histórico** en lenguaje natural. Eso es el RAG.

Empecé con uno básico y luego lo mejoré bastante. Cuatro cosas: embeddings con un modelo mejor, **e5**; un **filtrado inteligente** que detecta si preguntas por una zona o un tipo concreto y filtra antes de buscar; un **router** que manda las preguntas de "cuántos" a un cálculo exacto con pandas en vez de al modelo de lenguaje; y un **re-ranking** que reordena los resultados.

Y lo medí, que es lo importante. Mirad la tabla: el **MRR pasó de 0.78 a 1.0**. Eso quiere decir que ahora el primer resultado que recupera siempre es el relevante. Que es justo lo que necesitas, porque el modelo de lenguaje le hace más caso a lo primero.

---

## 10 · App

Todo esto se junta en la aplicación. Le di una estética de "radar de intercepción" a propósito, para que tuviera personalidad.

Tiene un **radar en vivo** que te dice el riesgo de la hora actual, con un punto por cada zona colocado según su posición geográfica real. Un **mapa** que puedes ver por zonas o como mapa de calor de los puntos reales. El **asistente** RAG. Y una pestaña de **estadísticas**.

*(Si hay demo en vivo: aquí cambio a la app y enseño el radar + una pregunta al chat.)*

---

## 11 · Stack

Rápido, las tecnologías. No me quería casar con una sola cosa, así que el proyecto toca varias áreas: procesamiento de lenguaje con la API de OpenAI, machine learning supervisado con XGBoost y Optuna, recuperación de información con e5 y ChromaDB, y visualización con Streamlit y Folium.

La idea era demostrar un poco de todo el abanico, no solo una técnica.

---

## 12 · Limitaciones

No quiero vender humo, así que las limitaciones. El histórico incluye los años de **COVID**, con confinamientos, así que esos patrones no son del todo extrapolables a hoy. Los datos están **sesgados por quién reporta** — no es un censo objetivo, es lo que la gente avisó. Y las probabilidades **no están calibradas**: sirven para ordenar zonas de más a menos riesgo, que es lo que hace la app, pero no me las creería como porcentaje exacto.

Como trabajo futuro: calibrar esas probabilidades, meter más variables como eventos locales, y dejar que el sistema vaya acumulando datos de 2026 con el proceso automático.

---

## 13 · Conclusión

Para cerrar: lo que creo que demuestra este proyecto es un **pipeline completo de IA**, de principio a fin. Coger un dato público, feo y sin estructura, y convertirlo en un producto que funciona.

Y que integra varias disciplinas — lenguaje, machine learning, recuperación de información — tomando decisiones con criterio: la validación temporal, las métricas honestas... Además es algo desplegable y que sigue aprendiendo solo.

---

## 14 · Cierre

Y ya está. Esto es Anti Hog. Muchas gracias, y quedo para las preguntas que tengáis.

---

# Preguntas probables (prepárate estas)

**¿Por qué XGBoost y no una red neuronal / deep learning?**
Con un dataset de este tamaño y features tabulares (día, hora, zona), los modelos de árboles como XGBoost rinden mejor que el deep learning y son más interpretables. Deep learning brillaría con muchísimos más datos.

**¿No es ilegal avisar de controles?**
En España compartir ubicaciones de controles es una zona gris pero no es delito — Waze lo hace. Aun así lo enfoco como análisis de patrones de datos públicos, con fin educativo, no operativo.

**El AUC bajó de 0.806 a 0.793 al pasar a 12 zonas. ¿No es peor?**
Es un intercambio consciente: pierdo 0.013 de AUC pero **duplico la resolución espacial**. La predicción es más útil aunque sea un pelín más difícil. Y sigue siendo un buen modelo.

**¿Por qué el F1 es tan bajo?**
Porque es un evento muy raro (1% de las celdas). El F1 con un threshold castiga eso. Pero la app no decide "sí o no", **ordena zonas por riesgo**, y para eso la métrica buena es el AUC, que es alto.

**¿Cómo sabes que el RAG mejoró de verdad?**
Lo medí con un set de evaluación: recall@8 y MRR, comparando la versión base contra la mejorada. El MRR pasó de 0.78 a 1.0. No es una sensación, es un número reproducible.

**¿Por qué validación temporal y no cross-validation?**
Porque es una serie temporal. Si hiciera k-fold normal, metería días del futuro en el entrenamiento y validaría con el pasado — estaría haciendo trampa y el resultado sería irrealmente bueno.

**¿Qué harías con más tiempo?**
Calibrar las probabilidades con regresión isotónica, ampliar el set de evaluación del RAG, y añadir variables externas como fiestas locales o eventos.
