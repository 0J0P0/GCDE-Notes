# Introducció
Volem prioritzar escalabilitat i tolerància a errors per sobre del rendiment. Per a fer-ho existeixen diverses arquitectures:
- **Client-servidor** (master slave) que pot ser *thin-client* o fat-client*.
- **Peer to peer**
  
Amb aquest objectiu es creen models de prgoramació guiats per les dades que tenen com objectiu aplicar una mateixa funció sobre un gran conjunt de dades. Es basen en particionar les dades i executar de manera concurrent la mateixa funcií a cada partició. 

## MapReduce
Pensat per a fer els mateixos càlculs sobre grups independents de dades. Primerament s'aplica un *map* en que les dades es filtren, ordenen o modifiquen seguit d'un *reduce* que agrega les dades. Ambdues estapes estan definides per a tuples`(key,value)`.
- **Map:** Aplica una funció $Map(k1,v1)\mapsto List(k2,v2)$. Que s'aplica a cada tuple key value de les dades. El domini d'entrada i sortida poden ser diferents.
- **Reduce:** és opcional. Agrega els resultats generats pel map. $Reduce(k2,List(v2))\mapsto List((k3,v3))$ L'entrada és valors que tenen la mateixa clau. Típicament produeix només un parell clau valor.

El framework MapReduce realitza les operacions de forma distribuida usant un conjunt de nodes. Primerament cada worker aplica el map a un conjunt de dades locals, després es realitza un *shuffle* per a redistribuir les dades basat en els output keys i finalment es realitza el reduce.

![](20230607155700.png =400x)

### WordCount
Cada document es divideix en words, cada word es transforma en una tupla (word,1) i es realitza un reduce per a sumar els valors de cada tupla amb la mateixa clau (mateix word).
```
function map(String name, String document):
    for each word w in document:
        emit(w,1)
function reduce(String word, Iterator partialCounts):
    sum = 0
    for each pc in partialCounts:
        sum += pc
    emit (word,sum)
```

### Pros i Contres
No necessàriament són la implementació més rpaìda però permeten haver de només escriure el Map i el Reduce. Hi ha un tradeoff entre cost computacional i cost de comunicació. 
Per a conjunts petits de dades i tasques senzilles normalment no és efectiu usar MapReduce ja que el overhead no compensa.
Són frameworks dissenyats amb la robustesa en ment; realitzant backups de les dades en diferents nodes.


# Apache Hadoop
És un MapReduce framework de codi obert basat en java i inspirat per Google. És escalable, resilient, d'alta availability i explota la localitat de les dades. Té 4 parts fonamentals; **Hadoop common**, **HDSFS**, **MapReduce**, el **YARN**.
L'arquitectura ideal és nodes amb discs locals que fan tant tasques de càlcul com gestió de les dades. D'aquesta manera els càlculs d'una partició es fan al mateix node on s'emmagametzen.

## HDFS
El Hadoop distributed file system s'encarrega de particionar i repartir les dades al cluster. Per a tenir tolerància a fallos usa replicació (normalment 3 còpies de les dades) o erasure coding. Aquest últim és més eficient a nivell d'emmagatzematge però més costos computacionalment.

Típicament un cluster està format per un **NameNode** que és el punt d'accés al cluster i l'encarregat del bookkeeping de les dades, implementar el namespace i assignar blocks als datanodes i de fer balanceig. El secondari està al dia del NameNode i actua com a backup preparat per entrar en acció, està en *standby*. Els **DataNodes** és on es guarden les dades i s'executen les tasques computacionals. Són data i worker nodes alhora normalment.

Per a poder treballar amb fitxers grans HDFS divideix les dades en blocs de igual mida preconfigurada (excepte el últim) que es reparteixen entre els nodes. Normalment 128MB i 3 rèpliques. El NameNode intenta prioritzar localitat de les dades.

HDFS es basa en el principi de *Write Once Read Many* **WORM**. Per a simplificar la coherència de les dades i augmentar el throughtput un cop un fitxer és creat o modificat no es pot modificar el que s'ha fet (però si afegir cosa al final). Només pot haver-hi un escriptor alhora. 

## MapReduce
Les fases el MapReduce job són:
- **split:** es particiona les dades i es repeteix pels nodes
- **map:** aplicar la funció de map a cada bloc
- **sort&shuffle:** el output dels mappers s'ordena i distribueix als reductors
- **reduce:** s'aplica la funció de reducció i es produeix un output

El split, sorting i shuffling els fa el framework i l'usuai ha d'implementar només el map i el reduce. Notem que el map i el reduce es poden fer en paral·lel a cada node ja que són independents. EL split no és el mateix que la repartició de les dades en blocs, el split depen de les keys, el map, etc.

A nivell d'implementació, les classes de key i value han d'implentar `Writable` i `WritableComparable` respectivament.


![](20230607155922.png =500x)


## Detalls de l'API

- Entrada de dades: ha d'implementar `getSplits` per a particionar l'entrada en `InputSplits`, que és la unitat de treball del mapper i també `createRecordReader` que és la instància d'una classe que donat un inputsplit obté un record (unitat de treball ed map). La classe per defecte és `TextInputFormat` i la configuració del job decideix la mida de cada InputSplit. Normalment el record és una linea on la key és la pisició dins del fitxer i el value el contingut de la linia.
- Operacions sobre les dades:
  - map: rep un conjunt de tuples `<key,value>` i produeix com a sortida un altre conjunt `<key,value>` que poden ser de tipus/valor diferent. Es llança un mapper per cada InputSplit.
  - combiner: funció opcional per agrupar valors de sortida amb la misma key dels mappers locals. Té sortida `<key,list(value)>`.
  - reducer: el resultat del map o combiner és ordenat i particionat per enviar als reducer. El nombre no depen de les dades sinó que és un paràmetre dehadoop. 

- Sortida de les dades: ha d'implementar `getRecordWriter`. Per defecte és TextOutputformat.

## Word Count
Usa les classes default. L'entrada és un fitxer de text, cada InputSplit un conjunt de linies i cada record 1 linia. 
- mapper: entrada `<nºlinea,text linea>`i sortida `<word,1>`.
- combiner: entrada `<paraula,list(nº ocurrencies)>` i sortida `<paraula, suma ocurrències>`.

ACABAR DE COCNRETAR
## YARN
FALTA MIRAR (NO ENTRA?)
# Spark
És una implementació de MapReduce que intenta oferir un millor rendiment que Hadoop. Entre d'altres evita escritpures intermitges a disc, usa caches, etc.


# Apache Spark

# Cassandra
És una base de dades escalable, eventualment consistent i distribuida d'estructures `key-value`. És de codi obert i actualment està gestionada per Apache.
## Sobre NoSQL
FALTA PER FER

## Arquitectura
Usa una arquitectura **peer-to-peer** de tal manera que tots els nodes intercanvien informació entre ells de manera continua. No hi ha nodes principals i els clients poden accedir a la DB desde des de qualsevol node. El node al qual el client es connecta actua com a **coordinador** entre aquest i la resta de nodes i determina quins nodes han de respondre a les consultes.

Un cluster conté un conjunt de nodes (virtuals o físics) que són el component bàsic de Cassandra i on es guarden les dades.

 
