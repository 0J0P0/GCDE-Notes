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
  - [Apache Spark](#apache-spark)
    - [Resilient Distributed Dataset (RDD)](#resilient-distributed-dataset-rdd)
    - [Inicializacion](#inicializacion)
  - [Transformaciones](#transformaciones)
  - [Acciones](#acciones)
  - [Task y etapas](#task-y-etapas)
  - [Spark SQL](#spark-sql)
    - [Transformaciones](#transformaciones-1)
    - [Accesos con lenguaje SQL](#accesos-con-lenguaje-sql)
  - [Cassandra](#cassandra)
  - [CAP theorem](#cap-theorem)

> El **cliente** es una aplicación informática o un ordenador que consume un servicio remoto en otro ordenador conocido como servidor.

> La **tolerancia a fallos** es la capacidad de un sistema a seguir continuando su funcionamiento cuando ocurre algún error.

> La **escalabilidad horizontal** hace referencia a una mejora proporcional en el tiempo de ejecución a medida que se añaden nuevos nodos de cálculo/almacenamiento.

> El nivel de **consistencia** corresponde al numero de nodos slaves que envian su respuesta al master.

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

**Ventajas:**
- Adaptarse dinamicamente a medida que cambian los requisitos. Pagar unicamente por los recursos que se usan.
- No es necesaria una inversion destinada a la infrestuctura desde el punto de vista del usuario.

**Desventajas:**
- Añade sobrecarga en el tiempo deejecucion de las aplicaciones.

## Arquitectura

**Nodos de control**:
- Gestionan los recursos de computacion.
- Monitorizan las actividades del cloud.
- Crear clusters virtuales para los usuarios.
- Crear puertas de entrada.

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
- Permite reservar recursos de computacion (CPU, memoria, almacenamiento, red) hardware bajo demanda de las necesidades del usuario (de manera dinamica).
- El usuario se encarga de la instalación del software necesario. No tiene que preocuparse por la administración del hardware.
  - **Ejemplo**: Amazon Web Services (AWS)

### Plataforma como servicio (PaaS)

- Ofrece una plataforma para desarrollar, ejecutar y administrar aplicaciones sin la complejidad de construir y mantener la infraestructura asociada.
- Junto con el software necesario para poder ejecutar las aplicaciones.
- El usuario no controla el hardware ni el software ya instalado.
  - **Ejemplo**: Google Cloud, Microsoft Azure

### Software como servicio (SaaS)

- Se proporciona software a traves de internet.
- Utilizacion de las aplicaciones que se ejecutan en el cloud.
- El usuario no tiene control sobre la infrestuctura (hardware o software) que se esta utilizando.
  - **Ejemplo**: Google Apps, Office 365

# Virtualizacion

**Virtualizacion**: creacion de una version virtual de un recurso o dispositivo, como un servidor, un sistema de almacenamiento, un sistema operativo o un recurso de red.

**Ventajas**: 
- Ilusion de multiples sistemas dedicados a cada usuario sobre un unico sistema fisico.
- Aislamiento de los recursos y facilita la tolerancia a fallos.
- Facilidad de portabilidad.
- Para facilitar la compartición de máquinas físicas (hardware)
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

- Se prioriza la escabilidad horizontal y la tolerancia a fallos sobre el alto rendimiento.
- Tipos de arquitecturas para sistemas distribuidos:
  - Cliente-Servidor: 
    - Modelo *thin client*: el cliente solo se encarga de la interfaz con el usuario, depende principalmente del servidor para el procesamiento.
    - Modelo *fat client* el cliente realiza la mayor parte posible del procesamiento, mientras que el servidor se encarga de la gestion de datos.
  - Peer-to-peer: Arquitectura descentralizada en la que todos los nodos son iguales y se comportan como clientes y servidores al mismo tiempo.

## MapReduce

**MapReduce**: modelo de programacion para procesar grandes volumenes de datos independientes.
- Mismos calculos sobre grupos independientes de datos.
- La sincronización entre las tareas se realiza a cargo del runtime. 

El procesado de datos se divide en dos partes:

**Map**: procesa una porción de datos de entrada y genera un conjunto de pares clave-valor intermedios.
- resultado parcial identificada por una clave que puede ser distinta a la clave de entrada.

**Reduce**: combina todos los valores asociados a una misma clave intermedia y genera un conjunto de valores de salida.
- resultado final identificado por una clave que puede ser distinta a la clave intermedia.
- opcional y ejecuta la agregacion de los resultados parciales.
- la entrada es una lista de valores generados por el map que tienen la misma clave.

## Apache Hadoop

Apache Hadoop es un framework de software diseñado para el procesamiento distribuido de grandes volúmenes de datos en clústeres de servidores. Hadoop se basa en el concepto de MapReduce, que divide las tareas en etapas de mapeo y reducción para procesar datos en paralelo. Además, Hadoop proporciona un sistema de archivos distribuido llamado Hadoop Distributed File System (HDFS) que permite el almacenamiento distribuido y la replicación de datos en el clúster. Hadoop es especialmente útil para procesar datos estructurados y no estructurados y es adecuado para cargas de trabajo batch.

- Tipo de arquitectura: Cliente-servidor (master-slave). El master recibe las peticiones de los clientes y las distribuye entre los slaves. Posibles cuellos de botella en el master.

- Se ejecuta en contenedores.
- Tolerancia a fallos (replicacion de datos y del master).
- Integrado con el sistema de ficheros HDFS (Hadoop Distributed File System).
  - HDFS: sistema de ficheros distribuido que permite el acceso a los datos de manera eficiente.
  - fichero dividido en bloques del mismo tamaño (128MB).
  - Bloques replicados en varios nodos para tolerancia a fallos. Muchas replicas --> mayor tolerancia a fallos pero mayor coste de almacenamiento y menor rendimiento.
  - Solo se pueden escribir una vez y solo se permite un escritor activo.
  - Ofrecen la vision de un único disco centralizado cuando en realidad la información está repartida entre varios nodos.
  - Namenode (replicas para tolerancia a fallos): Actúa como máster y almacena todos los metadatos necesarios para construir el sistema de ficheros a partir de los datos que almacenan los datanodes, es decir, almacena la estructura de directorios y de ficheros y los metadatos necesarios para componer cada fichero a partir de sus bloques. La localización de los bloques en el clúster la almacena en memoria RAM, a partir de la información que le proporcionan los datanodes al arrancar el sistema de archivos.
  - Datanode: Se pueden considerar esclavos, se limitan casi prácticamente a almacenar los bloques que componen cada fichero, así como, a proporcionarlos al namenode o a los clientes que lo solicitan.

**Componentes**:
- Master (Resource manager): uno por cada cluster. Recibe las peticiones de los clientes y organiza las tareas de procesamiento.
- Slaves (Node manager). Los slaves crean los contenedores para ejecutar los procesos.

**Pasos**:
- Dividir los datos de entrada en proporciones independientes. Particiones determinadas mediante *InputFormat*.   
  - El trozo de datos que se asigna a cada trozo se denomina *InputSplit*. Unidad de trabajo de un mapper.
  - Se necesita de *getSplit* para particionar la entrada en *InpusSplit*. 
  - *Record* es la unidad de trabajo de una funcion map. Corresponde a una linea.
- Procesar los datos en paralelo. (map task)
- Ordenar los resultados parciales y se envian a las tareas de reduce como datos de entrada.
  - Los datos de entrada y salida estan organizados en parejas clave-valor.

```apache
map()
# recibe como entrada un conjunto de datos (linea) y produce un conjunto de <key, value> intermedios (por cada linea).

combiner()
# funcion opcional para agrupar los valores de salida con la misma key de los mappers locales.

reducer()
# la salida del map() o combiner() se ordena y particiona para mandarselos a los reducers.
```

## Apache Spark

Apache Spark es un framework de procesamiento de datos de alto rendimiento y código abierto. A diferencia de Hadoop, que se basa principalmente en MapReduce, Spark utiliza un modelo de computación en memoria, lo que lo hace significativamente más rápido para ciertos tipos de operaciones. Spark ofrece una amplia gama de bibliotecas y herramientas para el procesamiento de datos en tiempo real, análisis de datos, aprendizaje automático (machine learning) y procesamiento de grafos. Spark puede integrarse con Hadoop y otros sistemas de almacenamiento, lo que le permite aprovechar datos de diversas fuentes.

- Tipo de arquitectura: Cliente-servidor (master-slave). El master recibe las peticiones de los clientes y las distribuye entre los slaves. Posibles cuellos de botella en el master.

- Driver (master): Analiza el codigo y pide los recursos necesarios. Asigna las tareas a los executors (slaves) y monitoriza su ejecucion.
  - Tolerancia a fallos (replicacion de datos y del master).

### Resilient Distributed Dataset (RDD)

**RDD**: conjunto de datos inmutable, distribuido y tolerante a fallos. Componente del core de Spark.

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

## Transformaciones

Una	transformación	toma	como	entrada	un	RDD	y	genera	como	resultado	uno	
o	más	RDD.	Además	se evalúan	de	manera	“perezosa”,	no	se	ejecutan	hasta	
que	el	planificador	se	encuentra	con	una	acción.

**Narrow**: todos los datos necesarios para calcular los registros de una particion. Se pueden resolver de manera local.
- `reduce`, `aggregate`

**Wide**: todos los datos necesarios para calcular los registros de una particion. Se necesitan datos de otras particiones.
- `filter`, `map`

## Acciones

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

## Cassandra

**Apache Cassandra:**
Apache Cassandra es una base de datos distribuida escalable y de alto rendimiento diseñada para manejar grandes volúmenes de datos en múltiples servidores. Cassandra está diseñada para ser altamente tolerante a fallos y ofrece una arquitectura descentralizada en la que todos los nodos de la base de datos son iguales y no hay un punto único de fallo. Cassandra se basa en el modelo de almacenamiento de columnas y proporciona una alta disponibilidad y escalabilidad lineal. Es especialmente adecuada para aplicaciones que requieren alta velocidad de escritura y acceso a datos distribuidos en múltiples ubicaciones geográficas.

- Tipo de arquitectura: Peer-to-peer. (*descentralizada*) Cualquier nodo puede recibir peticiones de los clientes. El nodo que recibe se convierte en el coordinador de la petición.
  - Redirige la parte de la petición correspondiente a cada nodo y espera a recibir las respuestas.

- Base de datos NoSQL.

**Ventajas:**
- Tolerancia a fallos al tener arquitectura peer-to-peer.
- Replicacion de datos.
- Soporte a datos no estructurados.
- Querys en tiempo real.

**Desventajas:**
- No hay un lenguaje estandar.
- Eficiencia de la query depende del modelo de datos. Dicta la forma del acceso a los datos.
- Las consultas de JOIN son muy costosas, ya que las diferentes particiones estan en diferentes nodos.

El particionado de datos pretende distribuir los datos de manera uniforme. La clave determina que nodo
se encarga de guardar ese record. Se hace aplicando una función de hash a la clave y asignando el nodo
que contiene el valor en su rango de tokens.

Organizacion de los datos en dos niveles: keyspace y tablas.
- Basada en la estructura clave-valor por columnas.
- **Primary key**: conjunto de uno o más atributos que identifica univocamente una *fila de datos*. Puede estar compuesta de una partition key y multiples clustering keys.
  - **Partition key**: parte de la *primary key* que se usa para identificar el nodo que contiene la fila. Tiene influencia sobre el balanceo de reparticion de datos.
  - **Clustering key**: parte de la clave (distinta de la *partition key*) que se usa para ordenar las filas dentro de una particion.
- **Partition**: conjunto de filas que comparten la misma partition key.

Es de interes:
- Acceder al minimo numero de nodos para obtener los datos. Leer el minimo numero de particiones por query.
- Distribucion de datos uniforme entre los nodos para maximizar el paralelismo.

Esrategia de replicacion: *SimpleStrategy* o *NetworkTopologyStrategy*.	

- **SimpleStrategy**: se replica en los siguientes nodos en el anillo.
- **NetworkTopologyStrategy**: se replica en los siguientes nodos en el anillo y en los siguientes nodos en el siguiente datacenter.

## CAP theorem

Enuncia que es imposible para un sistema de computación distribuido garantizar simultáneamente las tres propiedades siguientes:

- **Consistencia**: (Consistency) todos los nodos ven los mismos datos al mismo tiempo. Actualización de nodos con el último valor de la base de datos.
- **Disponibilidad** (Availability): todos los nodos responden a las peticiones.
- **Tolerancia de particiones** (Partition tolerance): el sistema sigue funcionando aunque se pierdan nodos.

Priozacion de **AP**. Se puede configurar la consistencia de las queries.
- Consistencia eventual: temporalmente los nodos pueden tener diferentes versiones de un dato
pero se garantiza que en algún momento todos los nodos tendrán la última versión