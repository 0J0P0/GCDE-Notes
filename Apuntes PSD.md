# Apuntes PSD

- [Apuntes PSD](#apuntes-psd)
- [Entornos de ejecución](#entornos-de-ejecución)
  - [Paradigmas de la computacion](#paradigmas-de-la-computacion)
  - [Tipos de sistemas](#tipos-de-sistemas)
  - [Requerimientos](#requerimientos)
- [Cloud computing](#cloud-computing)
  - [Arquitectura](#arquitectura)
  - [Servicios ofrecidos](#servicios-ofrecidos)
    - [Infrastuctura como servicio (IaaS)](#infrastuctura-como-servicio-iaas)
    - [Plataforma como servicio (PaaS)](#plataforma-como-servicio-paas)
    - [Software como servicio (SaaS)](#software-como-servicio-saas)
- [Virtualizacion](#virtualizacion)
  - [Emulacion](#emulacion)
  - [Virtualizacion a nivel hardware](#virtualizacion-a-nivel-hardware)
  - [Virtualizacion a nivel SO](#virtualizacion-a-nivel-so)
  - [Virtualizacion a nivel de aplicacion.](#virtualizacion-a-nivel-de-aplicacion)
- [Modelos de programación y runtimes](#modelos-de-programación-y-runtimes)
  - [MapReduce](#mapreduce)
  - [Apache Hadoop](#apache-hadoop)
    - [Resilient Distributed Dataset (RDD)](#resilient-distributed-dataset-rdd)
    - [Inicializacion](#inicializacion)
    - [Transformaciones](#transformaciones)
    - [Acciones](#acciones)
  - [Task y etapas](#task-y-etapas)
  - [Spark SQL](#spark-sql)
    - [Transformaciones](#transformaciones-1)
    - [Accesos con lenguaje SQL](#accesos-con-lenguaje-sql)
  - [Cositas](#cositas)

# Entornos de ejecución

Los entornos HPC se centran en acelerar al maximo el calculo de las aplicaciones, mientras que los entronos de computacion para el analisis de datos se centra en optimizar la utilizacion de los datos.

- La eficiencia en el uso de la CPU pierde relevancia.
- Enfoque en quitar el cuello de botella en la lectura de los datos. Para no detener el ritmo de la entrada y salida de los datos a la CPU. 
- El particionado de los datos influye en el paralelismo y el tiempo de gestion.
  - Los datos de entrada de una trasnformacion se particionan en bloques de datos y a cada particion se le crea una tarea.

## Paradigmas de la computacion

_Son modelos de computación..._

**Centralized computing**: todos los recursos estan en un unico sistema fisicos, gestionados por un unico sistema operativo.

**Parallel computing**: los procesadores puedenusar memoria fisica o pueden usar memoria distribuida. La comunicacion puede ser mediante memoria o mensajes.

**Distributed computing**: el sistema esta compuesto de ordenadores independientes entre si y la comunicacion se realiza mediante mensajes.

**Cloud computing**: Conjuntos disponibles a traves de internet (los recursos de computacion se entregan a traves de internet). Pueden estar construidos sobre recursos fisicos o virtuales.

## Tipos de sistemas

**Computer clusters**: conjunto de ordenadores interconectados que colaboran para ofrecer un unico recurso.
- Cada ordenador tiene su propio sistema operativo. Con una capa adicional que facilita la interaccion entre los procesos.

**Peer-to-peer Networks**: Conjunto de ordenadores interconectados en el que cada uno actua como cliente y servidor.
- La gestion esta distribuida entre todos. No hay un servidor central.


## Requerimientos

**Escalabilidad horizontal**: Si se añaden nuevos nodos de cálculo/almacenamiento, el rendimiento
debe mejorar de manera proporcional.

**Tolerancia a fallos**: capacidad de un sistema de continuar funcionando
cuando ocurre algún error. Si un nodo cae, los datos deben seguir accesibles.
- tiempo medio sin fallos alto y tiempo medio de recuperacion bajo.

# Cloud computing

**Cloud**: conjunto de recursos de computacion (hardware o software) ofrecios como servicios asignados dinamicamente y accesibles a traves de internet.
- Adaptarse dinamicamente a medida que cambian los requisitos. 
- No es necesaria una inversion destinada a la infrestuctura desde el punto de vista del usuario.

## Arquitectura

**Nodos de control**:
- Gestionan los recursos de computacion.
- Monitorizan las actividades del cloud.
- Crear clusters virtuales para los usuarios.
- crear puertas de entrada.

**Nodos de calculo** (nodos de trabajo):
- Ejecutan las tareas de los usuarios.
- Pueden ser compartidos por varios usuarios.

**Sistema subyacente**:
- Servidores independientes unidos a traves de una red.
- Discos locales.
- Redes optimizadas para el acceso a internet.

## Servicios ofrecidos

_Son modelos de servicio de cloud computing..._

### Infrastuctura como servicio (IaaS)

- Provee de recursos de computacion (CPU, memoria, almacenamiento, red) hardware.
- El usuario es responsable de la gestion de los recursos. No controla el hardware ni el software ya instalado.

### Plataforma como servicio (PaaS)

- Ofrece una plataforma para desarrollar, ejecutar y administrar aplicaciones sin la complejidad de construir y mantener la infraestructura asociada.
- Junto con el software necesario para poder ejecutar las aplicaciones.
- El usuario es responsable de la gestion de los recursos. No controla el hardware ni el software ya instalado.

### Software como servicio (SaaS)

- Se proporciona software a traves de internet.
- Utilizacion de las aplicaciones que se ejecutan en el cloud.
- El usuario no tiene control sobre la infrestuctura (hardware o software) que se esta utilizando.

# Virtualizacion

**Virtualizacion**: creacion de una version virtual de un recurso o dispositivo, como un servidor, un sistema de almacenamiento, un sistema operativo o un recurso de red.

**Ventajas**: 
- Ilusion de multiples sistemas dedicados sobre un unico sistema fisico.
- Aislamiento de los recursos y facilita la tolerancia a fallos.
- Facilidad de portabilidad.
- Para facilitar la compartición de máquinas físicas
- Para facilitar la asignación dinámica de recursos
- Para facilitar la gestión de los entornos de ejecución (instalación y configuración de
software)

## Emulacion

**Emulacion**: creacion de un entorno de ejecucion que imita el comportamiento de un sistema diferente al que se esta ejecutando.

## Virtualizacion a nivel hardware

Permite crear una abstracción del sistema completo (HW+SW), permitiendo que un SO guest se ejecute de manera aislada sobre un sistema nativo (host)

## Virtualizacion a nivel SO

Se basa en replicar únicamente el espacio a nivel de usuario compartiendo un único SO: containers.

## Virtualizacion a nivel de aplicacion.

Se basa en replicar el espacio de direcciones de una aplicación, permitiendo que se ejecute de manera aislada sobre un sistema nativo (host).

# Modelos de programación y runtimes

**Apache Hadoop:**
Apache Hadoop es un framework de software diseñado para el procesamiento distribuido de grandes volúmenes de datos en clústeres de servidores. Hadoop se basa en el concepto de MapReduce, que divide las tareas en etapas de mapeo y reducción para procesar datos en paralelo. Además, Hadoop proporciona un sistema de archivos distribuido llamado Hadoop Distributed File System (HDFS) que permite el almacenamiento distribuido y la replicación de datos en el clúster. Hadoop es especialmente útil para procesar datos estructurados y no estructurados y es adecuado para cargas de trabajo batch.

**Apache Spark:**
Apache Spark es un framework de procesamiento de datos de alto rendimiento y código abierto. A diferencia de Hadoop, que se basa principalmente en MapReduce, Spark utiliza un modelo de computación en memoria, lo que lo hace significativamente más rápido para ciertos tipos de operaciones. Spark ofrece una amplia gama de bibliotecas y herramientas para el procesamiento de datos en tiempo real, análisis de datos, aprendizaje automático (machine learning) y procesamiento de grafos. Spark puede integrarse con Hadoop y otros sistemas de almacenamiento, lo que le permite aprovechar datos de diversas fuentes.

**Apache Cassandra:**
Apache Cassandra es una base de datos distribuida escalable y de alto rendimiento diseñada para manejar grandes volúmenes de datos en múltiples servidores. Cassandra está diseñada para ser altamente tolerante a fallos y ofrece una arquitectura descentralizada en la que todos los nodos de la base de datos son iguales y no hay un punto único de fallo. Cassandra se basa en el modelo de almacenamiento de columnas y proporciona una alta disponibilidad y escalabilidad lineal. Es especialmente adecuada para aplicaciones que requieren alta velocidad de escritura y acceso a datos distribuidos en múltiples ubicaciones geográficas.

- Se prioriza la escabilidad horizontal y la tolerancia a fallos sobre el alto rendimiento.

## MapReduce

**MapReduce**: modelo de programacion para procesar grandes volumenes de datos de manera distribuida y paralela.
- Mismos calculos sobre grupos independientes de datos.

El procesado de datos se divide en dos partes:

**Map**: procesa los datos de entrada y genera un conjunto de pares clave-valor intermedios.
- resultado parcial identificadapor una clave que puede ser distinta a la clave de entrada.

**Reduce**: combina todos los valores asociados a una misma clave intermedia y genera un conjunto de valores de salida.
- resultado final identificado por una clave que puede ser distinta a la clave intermedia.
- opcional y ejecuta la agregacion de los resultados parciales.
- la entrada es una lista de valores generados por el map que tienen la misma clave.

## Apache Hadoop

**Apache Hadoop**: framework de software que permite el procesamiento distribuido de grandes volumenes de datos a traves de clusters de ordenadores usando modelos de programacion simples.

*Driver* es el nodo master. (analiza el codigo y pide los recursos necesarios para ejecutarlo) (planifica las tasks y monitoriza la ejecucion para detectar y corregir errores).

*Executors* son los slaves que se encargan de ejecutar las tasks.

- Para ejecutar aplicaciones MapReduce.
- Tolerancia a fallos.
- Integrado con el sistema de ficheros HDFS (Hadoop Distributed File System).
  - HDFS: sistema de ficheros distribuido que permite el acceso a los datos de manera eficiente.
  - fichero dividido en bloques del mismo tamaño.
  - Bloques replicados en varios nodos (slave).
  - Solo se pueden escribir una vez y solo se permite un escritor activo.
- Arquitectura ideal: nodos con discos locales que hae tanto tareas de calculo como de gestion de datos.

**Componentes**:
- Master y slaves. Los slaves crean los contenedores para ejecutar los procesos.

- Ofrece a los usuarios la vision de un solo disco centralizado cuando en realidad la informacion esta repartida entre varios nodos.

**Pasos**:
- Dividir los datos de entrada en proporciones independientes.
- Procesar los datos en paralelo. (map task)
- Ordenar los resultados parciales y se envian a las tareas de reduce como datos de entrada.
  - Los datos de entrada y salida estan organizados en parejas clave-valor.
  - Es wide porque datos con la misma key pueden estar distribuidos por más de un
nodo. 


```apache
map()
# recibe como entrada un conjunto de datos (linea) y produce un conjunto de <key, value> intermedios (por cada linea).

combiner()
# funcion opcional para agrupar los valores de salida con la misma key de los mappers locales.

reducer()
# la salida del map() o combiner() se ordena y particiona para mandarselos a los reducers.
```

### Resilient Distributed Dataset (RDD)

**RDD**: conjunto de datos inmutable, distribuido y tolerante a fallos.
- Solo de lectura.
- Conjunto de records particionados y distribuidos entre los nodos del cluster. Para poder procesarlos en paralelo.
- Se pueden crear a partir de datos en HDFS o de otros RDDs.
- Creacion a partir de colecciones en memoria (local) o de ficheros en HDFS (distribuido).

### Inicializacion

```python
from pyspark import SparkContext
sc = SparkContext("local", "Nombre") # para ejecucion local secuencial --> el fichero se cargara a memoria
sc = SparkContext("local[*]", "Nombre") # para ejecucion local paralela (usando los cores del nodo)
sc = SparkContext("spark://master:7077", "Nombre") # para ejecucion distribuida (url del master)

# o bien
conf = SparkConf().setAppName("Nombre").setMaster("local") # para ejecucion local secuencial
sc = SparkContext(conf=conf)
```

### Transformaciones

Una	transformación	toma	como	entrada	un	RDD	y	genera	como	resultado	uno	
o	más	RDD.	Además	se evalúan	de	manera	“perezosa”,	no	se	ejecutan	hasta	
que	el	planificador	se	encuentra	con	una	acción.

**Narrow**: todos los datos necesarios para calcular los registros de una particion. Se pueden resolver de manera local.

**Wide**: todos los datos necesarios para calcular los registros de una particion. Se necesitan datos de otras particiones.

### Acciones

Generan un resultado que se guarda en almacenamiento o se muestra en pantalla.

Una	acción	trabaja	sobre	un	RDD	pero	la	salida	ya	no	es	otro	RDD	sino	que	
puede ser,	por	ejemplo,	la	escritura	en	un	dispositivo	o	el	envío	de	un	resultado	
al	driver de	la	aplicación

- Dan por finalizada una secuencia de transformaciones y desencadenan su ejecucion.

## Task y etapas

El runtime se encarga de implementar las operaciones de sincronización entre
las tareas, no es necesario que el programador la invoque explícitamente.

- *Logical execution plan*
- Un grafo dirijiudo aciclico (DAG) de operaciones que se ejecutan en paralelo.
- Creado por el driver 
- Expresa las dependencias entre los RDD. Las aristas son las transformaciones y los nodos son los RDD.

Cuando el driver ejecuta un action, se crea un DAG de tareas que se ejecutan en paralelo. DAGscheduler convierte el DAG de RDD en un DAG de tareas basado en etapas.

- *Physical execution plan*
- **Etapa**: conjunto de transformaciones que se pueden resolver de manera local. (Solo usa datos de una transformacion) (Narrow)

## Spark SQL

Se debe crear una sesion de Spark para poder usar Spark SQL.

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Nombre").getOrCreate()

df = spark.read.load("file.csv", format="csv", sep=",", inferSchema="true", header="true") # desde un fichero

# Desde memoria
df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "letra"]) # desde una lista, diccionario o dataframe
```

### Transformaciones

```python
df.select("nombre_columna") # seleccionar columnas
df.filter(df["nombre_columna"] > 0) # filtrar filas
df.groupBy("nombre_columna").count() # agrupar por columnas
df.sort(df["nombre_columna"].desc()) # ordenar por columnas
```

### Accesos con lenguaje SQL

```python
df.createOrReplaceTempView("nombre_tabla") # crear una vista temporal
spark.sql("SELECT * FROM nombre_tabla") # ejecutar una consulta SQL
```

## Cositas

Un fichero .json tiene un conjunto de records. Cada record es un diccionario que es el dato de entrada de la transformacion.
