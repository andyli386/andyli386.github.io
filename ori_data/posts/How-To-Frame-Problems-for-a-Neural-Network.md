---
title: How To "Frame Problems" for a Neural Network
date: 2017-02-12 21:01:02
tags:
- 纳米学位 
- DLND
- 机器学习
---

# Sentiment Classification & How To "Frame Problems" for a Neural Network

by Andrew Trask

- **Twitter**: @iamtrask
- **Blog**: http://iamtrask.github.io

### What You Should Already Know

- neural networks, forward and back-propagation
- stochastic gradient descent
- mean squared error
- and train/test splits

### Where to Get Help if You Need it
- Re-watch previous Udacity Lectures
- Leverage the recommended Course Reading Material - [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) (40% Off: **traskud17**)
- Shoot me a tweet @iamtrask
<!--more-->

### Tutorial Outline:

- Intro: The Importance of "Framing a Problem"


- Curate a Dataset
- Developing a "Predictive Theory"
- **PROJECT 1**: Quick Theory Validation


- Transforming Text to Numbers
- **PROJECT 2**: Creating the Input/Output Data


- Putting it all together in a Neural Network
- **PROJECT 3**: Building our Neural Network


- Understanding Neural Noise
- **PROJECT 4**: Making Learning Faster by Reducing Noise


- Analyzing Inefficiencies in our Network
- **PROJECT 5**: Making our Network Train and Run Faster


- Further Noise Reduction
- **PROJECT 6**: Reducing Noise by Strategically Reducing the Vocabulary


- Analysis: What's going on in the weights?

# Lesson: Curate a Dataset


```python
def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()
```


```python
len(reviews)
```




    25000




```python
reviews[0]
```




    'bromwell high is a cartoon comedy . it ran at the same time as some other programs about school life  such as  teachers  . my   years in the teaching profession lead me to believe that bromwell high  s satire is much closer to reality than is  teachers  . the scramble to survive financially  the insightful students who can see right through their pathetic teachers  pomp  the pettiness of the whole situation  all remind me of the schools i knew and their students . when i saw the episode in which a student repeatedly tried to burn down the school  i immediately recalled . . . . . . . . . at . . . . . . . . . . high . a classic line inspector i  m here to sack one of your teachers . student welcome to bromwell high . i expect that many adults of my age think that bromwell high is far fetched . what a pity that it isn  t   '




```python
labels[0]
```




    'POSITIVE'



# Lesson: Develop a Predictive Theory


```python
print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)
```

    labels.txt 	 : 	 reviews.txt
    
    NEGATIVE	:	this movie is terrible but it has some good effects .  ...
    POSITIVE	:	adrian pasdar is excellent is this film . he makes a fascinating woman .  ...
    NEGATIVE	:	comment this movie is impossible . is terrible  very improbable  bad interpretat...
    POSITIVE	:	excellent episode movie ala pulp fiction .  days   suicides . it doesnt get more...
    NEGATIVE	:	if you haven  t seen this  it  s terrible . it is pure trash . i saw this about ...
    POSITIVE	:	this schiffer guy is a real genius  the movie is of excellent quality and both e...


# Project 1: Quick Theory Validation


```python
from collections import Counter
import numpy as np
```


```python
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()
```


```python
for i in range(len(reviews)):
    if(labels[i] == 'POSITIVE'):
        for word in reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1
```


```python
positive_counts.most_common()
```




    [('', 550468),
     ('the', 173324),
     ('.', 159654),
     ('and', 89722),
     ('a', 83688),
     ('of', 76855),
     ('to', 66746),
     ('is', 57245),
     ('in', 50215),
     ('br', 49235),
     ('it', 48025),
     ('i', 40743),
     ('that', 35630),
     ('this', 35080),
     ('s', 33815),
     ('as', 26308),
     ('with', 23247),
     ('for', 22416),
     ('was', 21917),
     ('film', 20937),
     ('but', 20822),
     ('movie', 19074),
     ('his', 17227),
     ('on', 17008),
     ('you', 16681),
     ('he', 16282),
     ('are', 14807),
     ('not', 14272),
     ('t', 13720),
     ('one', 13655),
     ('have', 12587),
     ('be', 12416),
     ('by', 11997),
     ('all', 11942),
     ('who', 11464),
     ('an', 11294),
     ('at', 11234),
     ('from', 10767),
     ('her', 10474),
     ('they', 9895),
     ('has', 9186),
     ('so', 9154),
     ('like', 9038),
     ('about', 8313),
     ('very', 8305),
     ('out', 8134),
     ('there', 8057),
     ('she', 7779),
     ('what', 7737),
     ('or', 7732),
     ('good', 7720),
     ('more', 7521),
     ('when', 7456),
     ('some', 7441),
     ('if', 7285),
     ('just', 7152),
     ('can', 7001),
     ('story', 6780),
     ('time', 6515),
     ('my', 6488),
     ('great', 6419),
     ('well', 6405),
     ('up', 6321),
     ('which', 6267),
     ('their', 6107),
     ('see', 6026),
     ('also', 5550),
     ('we', 5531),
     ('really', 5476),
     ('would', 5400),
     ('will', 5218),
     ('me', 5167),
     ('had', 5148),
     ('only', 5137),
     ('him', 5018),
     ('even', 4964),
     ('most', 4864),
     ('other', 4858),
     ('were', 4782),
     ('first', 4755),
     ('than', 4736),
     ('much', 4685),
     ('its', 4622),
     ('no', 4574),
     ('into', 4544),
     ('people', 4479),
     ('best', 4319),
     ('love', 4301),
     ('get', 4272),
     ('how', 4213),
     ('life', 4199),
     ('been', 4189),
     ('because', 4079),
     ('way', 4036),
     ('do', 3941),
     ('made', 3823),
     ('films', 3813),
     ('them', 3805),
     ('after', 3800),
     ('many', 3766),
     ('two', 3733),
     ('too', 3659),
     ('think', 3655),
     ('movies', 3586),
     ('characters', 3560),
     ('character', 3514),
     ('don', 3468),
     ('man', 3460),
     ('show', 3432),
     ('watch', 3424),
     ('seen', 3414),
     ('then', 3358),
     ('little', 3341),
     ('still', 3340),
     ('make', 3303),
     ('could', 3237),
     ('never', 3226),
     ('being', 3217),
     ('where', 3173),
     ('does', 3069),
     ('over', 3017),
     ('any', 3002),
     ('while', 2899),
     ('know', 2833),
     ('did', 2790),
     ('years', 2758),
     ('here', 2740),
     ('ever', 2734),
     ('end', 2696),
     ('these', 2694),
     ('such', 2590),
     ('real', 2568),
     ('scene', 2567),
     ('back', 2547),
     ('those', 2485),
     ('though', 2475),
     ('off', 2463),
     ('new', 2458),
     ('your', 2453),
     ('go', 2440),
     ('acting', 2437),
     ('plot', 2432),
     ('world', 2429),
     ('scenes', 2427),
     ('say', 2414),
     ('through', 2409),
     ('makes', 2390),
     ('better', 2381),
     ('now', 2368),
     ('work', 2346),
     ('young', 2343),
     ('old', 2311),
     ('ve', 2307),
     ('find', 2272),
     ('both', 2248),
     ('before', 2177),
     ('us', 2162),
     ('again', 2158),
     ('series', 2153),
     ('quite', 2143),
     ('something', 2135),
     ('cast', 2133),
     ('should', 2121),
     ('part', 2098),
     ('always', 2088),
     ('lot', 2087),
     ('another', 2075),
     ('actors', 2047),
     ('director', 2040),
     ('family', 2032),
     ('between', 2016),
     ('own', 2016),
     ('m', 1998),
     ('may', 1997),
     ('same', 1972),
     ('role', 1967),
     ('watching', 1966),
     ('every', 1954),
     ('funny', 1953),
     ('doesn', 1935),
     ('performance', 1928),
     ('few', 1918),
     ('bad', 1907),
     ('look', 1900),
     ('re', 1884),
     ('why', 1855),
     ('things', 1849),
     ('times', 1832),
     ('big', 1815),
     ('however', 1795),
     ('actually', 1790),
     ('action', 1789),
     ('going', 1783),
     ('bit', 1757),
     ('comedy', 1742),
     ('down', 1740),
     ('music', 1738),
     ('must', 1728),
     ('take', 1709),
     ('saw', 1692),
     ('long', 1690),
     ('right', 1688),
     ('fun', 1686),
     ('fact', 1684),
     ('excellent', 1683),
     ('around', 1674),
     ('didn', 1672),
     ('without', 1671),
     ('thing', 1662),
     ('thought', 1639),
     ('got', 1635),
     ('each', 1630),
     ('day', 1614),
     ('feel', 1597),
     ('seems', 1596),
     ('come', 1594),
     ('done', 1586),
     ('beautiful', 1580),
     ('especially', 1572),
     ('played', 1571),
     ('almost', 1566),
     ('want', 1562),
     ('yet', 1556),
     ('give', 1553),
     ('pretty', 1549),
     ('last', 1543),
     ('since', 1519),
     ('different', 1504),
     ('although', 1501),
     ('gets', 1490),
     ('true', 1487),
     ('interesting', 1481),
     ('job', 1470),
     ('enough', 1455),
     ('our', 1454),
     ('shows', 1447),
     ('horror', 1441),
     ('woman', 1439),
     ('tv', 1400),
     ('probably', 1398),
     ('father', 1395),
     ('original', 1393),
     ('girl', 1390),
     ('point', 1379),
     ('plays', 1378),
     ('wonderful', 1372),
     ('far', 1358),
     ('course', 1358),
     ('john', 1350),
     ('rather', 1340),
     ('isn', 1328),
     ('ll', 1326),
     ('later', 1324),
     ('dvd', 1324),
     ('whole', 1310),
     ('war', 1310),
     ('d', 1307),
     ('found', 1306),
     ('away', 1306),
     ('screen', 1305),
     ('nothing', 1300),
     ('year', 1297),
     ('once', 1296),
     ('hard', 1294),
     ('together', 1280),
     ('set', 1277),
     ('am', 1277),
     ('having', 1266),
     ('making', 1265),
     ('place', 1263),
     ('might', 1260),
     ('comes', 1260),
     ('sure', 1253),
     ('american', 1248),
     ('play', 1245),
     ('kind', 1244),
     ('perfect', 1242),
     ('takes', 1242),
     ('performances', 1237),
     ('himself', 1230),
     ('worth', 1221),
     ('everyone', 1221),
     ('anyone', 1214),
     ('actor', 1203),
     ('three', 1201),
     ('wife', 1196),
     ('classic', 1192),
     ('goes', 1186),
     ('ending', 1178),
     ('version', 1168),
     ('star', 1149),
     ('enjoy', 1146),
     ('book', 1142),
     ('nice', 1132),
     ('everything', 1128),
     ('during', 1124),
     ('put', 1118),
     ('seeing', 1111),
     ('least', 1102),
     ('house', 1100),
     ('high', 1095),
     ('watched', 1094),
     ('loved', 1087),
     ('men', 1087),
     ('night', 1082),
     ('anything', 1075),
     ('believe', 1071),
     ('guy', 1071),
     ('top', 1063),
     ('amazing', 1058),
     ('hollywood', 1056),
     ('looking', 1053),
     ('main', 1044),
     ('definitely', 1043),
     ('gives', 1031),
     ('home', 1029),
     ('seem', 1028),
     ('episode', 1023),
     ('audience', 1020),
     ('sense', 1020),
     ('truly', 1017),
     ('special', 1011),
     ('second', 1009),
     ('short', 1009),
     ('fan', 1009),
     ('mind', 1005),
     ('human', 1001),
     ('recommend', 999),
     ('full', 996),
     ('black', 995),
     ('help', 991),
     ('along', 989),
     ('trying', 987),
     ('small', 986),
     ('death', 985),
     ('friends', 981),
     ('remember', 974),
     ('often', 970),
     ('said', 966),
     ('favorite', 962),
     ('heart', 959),
     ('early', 957),
     ('left', 956),
     ('until', 955),
     ('script', 954),
     ('let', 954),
     ('maybe', 937),
     ('today', 936),
     ('live', 934),
     ('less', 934),
     ('moments', 933),
     ('others', 929),
     ('brilliant', 926),
     ('shot', 925),
     ('liked', 923),
     ('become', 916),
     ('won', 915),
     ('used', 910),
     ('style', 907),
     ('mother', 895),
     ('lives', 894),
     ('came', 893),
     ('stars', 890),
     ('cinema', 889),
     ('looks', 885),
     ('perhaps', 884),
     ('read', 882),
     ('enjoyed', 879),
     ('boy', 875),
     ('drama', 873),
     ('highly', 871),
     ('given', 870),
     ('playing', 867),
     ('use', 864),
     ('next', 859),
     ('women', 858),
     ('fine', 857),
     ('effects', 856),
     ('kids', 854),
     ('entertaining', 853),
     ('need', 852),
     ('line', 850),
     ('works', 848),
     ('someone', 847),
     ('mr', 836),
     ('simply', 835),
     ('picture', 833),
     ('children', 833),
     ('face', 831),
     ('keep', 831),
     ('friend', 831),
     ('dark', 830),
     ('overall', 828),
     ('certainly', 828),
     ('minutes', 827),
     ('wasn', 824),
     ('history', 822),
     ('finally', 820),
     ('couple', 816),
     ('against', 815),
     ('son', 809),
     ('understand', 808),
     ('lost', 807),
     ('michael', 805),
     ('else', 801),
     ('throughout', 798),
     ('fans', 797),
     ('city', 792),
     ('reason', 789),
     ('written', 787),
     ('production', 787),
     ('several', 784),
     ('school', 783),
     ('based', 781),
     ('rest', 781),
     ('try', 780),
     ('dead', 776),
     ('hope', 775),
     ('strong', 768),
     ('white', 765),
     ('tell', 759),
     ('itself', 758),
     ('half', 753),
     ('person', 749),
     ('sometimes', 746),
     ('past', 744),
     ('start', 744),
     ('genre', 743),
     ('beginning', 739),
     ('final', 739),
     ('town', 738),
     ('art', 734),
     ('humor', 732),
     ('game', 732),
     ('yes', 731),
     ('idea', 731),
     ('late', 730),
     ('becomes', 729),
     ('despite', 729),
     ('able', 726),
     ('case', 726),
     ('money', 723),
     ('child', 721),
     ('completely', 721),
     ('side', 719),
     ('camera', 716),
     ('getting', 714),
     ('instead', 712),
     ('soon', 702),
     ('under', 700),
     ('viewer', 699),
     ('age', 697),
     ('days', 696),
     ('stories', 696),
     ('felt', 694),
     ('simple', 694),
     ('roles', 693),
     ('video', 688),
     ('name', 683),
     ('either', 683),
     ('doing', 677),
     ('turns', 674),
     ('wants', 671),
     ('close', 671),
     ('title', 669),
     ('wrong', 668),
     ('went', 666),
     ('james', 665),
     ('evil', 659),
     ('budget', 657),
     ('episodes', 657),
     ('relationship', 655),
     ('fantastic', 653),
     ('piece', 653),
     ('david', 651),
     ('turn', 648),
     ('murder', 646),
     ('parts', 645),
     ('brother', 644),
     ('absolutely', 643),
     ('head', 643),
     ('experience', 642),
     ('eyes', 641),
     ('sex', 638),
     ('direction', 637),
     ('called', 637),
     ('directed', 636),
     ('lines', 634),
     ('behind', 633),
     ('sort', 632),
     ('actress', 631),
     ('lead', 630),
     ('oscar', 628),
     ('including', 627),
     ('example', 627),
     ('known', 625),
     ('musical', 625),
     ('chance', 621),
     ('score', 620),
     ('already', 619),
     ('feeling', 619),
     ('hit', 619),
     ('voice', 615),
     ('moment', 612),
     ('living', 612),
     ('low', 610),
     ('supporting', 610),
     ('ago', 609),
     ('themselves', 608),
     ('reality', 605),
     ('hilarious', 605),
     ('jack', 604),
     ('told', 603),
     ('hand', 601),
     ('quality', 600),
     ('moving', 600),
     ('dialogue', 600),
     ('song', 599),
     ('happy', 599),
     ('matter', 598),
     ('paul', 598),
     ('light', 594),
     ('future', 593),
     ('entire', 592),
     ('finds', 591),
     ('gave', 589),
     ('laugh', 587),
     ('released', 586),
     ('expect', 584),
     ('fight', 581),
     ('particularly', 580),
     ('cinematography', 579),
     ('police', 579),
     ('whose', 578),
     ('type', 578),
     ('sound', 578),
     ('view', 573),
     ('enjoyable', 573),
     ('number', 572),
     ('romantic', 572),
     ('husband', 572),
     ('daughter', 572),
     ('documentary', 571),
     ('self', 570),
     ('superb', 569),
     ('modern', 569),
     ('took', 569),
     ('robert', 569),
     ('mean', 566),
     ('shown', 563),
     ('coming', 561),
     ('important', 560),
     ('king', 559),
     ('leave', 559),
     ('change', 558),
     ('somewhat', 555),
     ('wanted', 555),
     ('tells', 554),
     ('events', 552),
     ('run', 552),
     ('career', 552),
     ('country', 552),
     ('heard', 550),
     ('season', 550),
     ('greatest', 549),
     ('girls', 549),
     ('etc', 547),
     ('care', 546),
     ('starts', 545),
     ('english', 542),
     ('killer', 541),
     ('tale', 540),
     ('guys', 540),
     ('totally', 540),
     ('animation', 540),
     ('usual', 539),
     ('miss', 535),
     ('opinion', 535),
     ('easy', 531),
     ('violence', 531),
     ('songs', 530),
     ('british', 528),
     ('says', 526),
     ('realistic', 525),
     ('writing', 524),
     ('writer', 522),
     ('act', 522),
     ('comic', 521),
     ('thriller', 519),
     ('television', 517),
     ('power', 516),
     ('ones', 515),
     ('kid', 514),
     ('york', 513),
     ('novel', 513),
     ('alone', 512),
     ('problem', 512),
     ('attention', 509),
     ('involved', 508),
     ('kill', 507),
     ('extremely', 507),
     ('seemed', 506),
     ('hero', 505),
     ('french', 505),
     ('rock', 504),
     ('stuff', 501),
     ('wish', 499),
     ('begins', 498),
     ('taken', 497),
     ('sad', 497),
     ('ways', 496),
     ('richard', 495),
     ('knows', 494),
     ('atmosphere', 493),
     ('similar', 491),
     ('surprised', 491),
     ('taking', 491),
     ('car', 491),
     ('george', 490),
     ('perfectly', 490),
     ('across', 489),
     ('team', 489),
     ('eye', 489),
     ('sequence', 489),
     ('room', 488),
     ('due', 488),
     ('among', 488),
     ('serious', 488),
     ('powerful', 488),
     ('strange', 487),
     ('order', 487),
     ('cannot', 487),
     ('b', 487),
     ('beauty', 486),
     ('famous', 485),
     ('happened', 484),
     ('tries', 484),
     ('herself', 484),
     ('myself', 484),
     ('class', 483),
     ('four', 482),
     ('cool', 481),
     ('release', 479),
     ('anyway', 479),
     ('theme', 479),
     ('opening', 478),
     ('entertainment', 477),
     ('slow', 475),
     ('ends', 475),
     ('unique', 475),
     ('exactly', 475),
     ('easily', 474),
     ('level', 474),
     ('o', 474),
     ('red', 474),
     ('interest', 472),
     ('happen', 471),
     ('crime', 470),
     ('viewing', 468),
     ('sets', 467),
     ('memorable', 467),
     ('stop', 466),
     ('group', 466),
     ('problems', 463),
     ('dance', 463),
     ('working', 463),
     ('sister', 463),
     ('message', 463),
     ('knew', 462),
     ('mystery', 461),
     ('nature', 461),
     ('bring', 460),
     ('believable', 459),
     ('thinking', 459),
     ('brought', 459),
     ('mostly', 458),
     ('disney', 457),
     ('couldn', 457),
     ('society', 456),
     ('lady', 455),
     ('within', 455),
     ('blood', 454),
     ('parents', 453),
     ('upon', 453),
     ('viewers', 453),
     ('meets', 452),
     ('form', 452),
     ('peter', 452),
     ('tom', 452),
     ('usually', 452),
     ('soundtrack', 452),
     ('local', 450),
     ('certain', 448),
     ('follow', 448),
     ('whether', 447),
     ('possible', 446),
     ('emotional', 445),
     ('killed', 444),
     ('above', 444),
     ('de', 444),
     ('god', 443),
     ('middle', 443),
     ('needs', 442),
     ('happens', 442),
     ('flick', 442),
     ('masterpiece', 441),
     ('period', 440),
     ('major', 440),
     ('named', 439),
     ('haven', 439),
     ('particular', 438),
     ('th', 438),
     ('earth', 437),
     ('feature', 437),
     ('stand', 436),
     ('words', 435),
     ('typical', 435),
     ('elements', 433),
     ('obviously', 433),
     ('romance', 431),
     ('jane', 430),
     ('yourself', 427),
     ('showing', 427),
     ('brings', 426),
     ('fantasy', 426),
     ('guess', 423),
     ('america', 423),
     ('unfortunately', 422),
     ('huge', 422),
     ('indeed', 421),
     ('running', 421),
     ('talent', 420),
     ('stage', 419),
     ('started', 418),
     ('leads', 417),
     ('sweet', 417),
     ('japanese', 417),
     ('poor', 416),
     ('deal', 416),
     ('incredible', 413),
     ('personal', 413),
     ('fast', 412),
     ('became', 410),
     ('deep', 410),
     ('hours', 409),
     ('giving', 408),
     ('nearly', 408),
     ('dream', 408),
     ('clearly', 407),
     ('turned', 407),
     ('obvious', 406),
     ('near', 406),
     ('cut', 405),
     ('surprise', 405),
     ('era', 404),
     ('body', 404),
     ('hour', 403),
     ('female', 403),
     ('five', 403),
     ('note', 399),
     ('learn', 398),
     ('truth', 398),
     ('except', 397),
     ('feels', 397),
     ('match', 397),
     ('tony', 397),
     ('filmed', 394),
     ('clear', 394),
     ('complete', 394),
     ('street', 393),
     ('eventually', 393),
     ('keeps', 393),
     ('older', 393),
     ('lots', 393),
     ('buy', 392),
     ('william', 391),
     ('stewart', 391),
     ('fall', 390),
     ('joe', 390),
     ('meet', 390),
     ('unlike', 389),
     ('talking', 389),
     ('shots', 389),
     ('rating', 389),
     ('difficult', 389),
     ('dramatic', 388),
     ('means', 388),
     ('situation', 386),
     ('wonder', 386),
     ('present', 386),
     ('appears', 386),
     ('subject', 386),
     ('comments', 385),
     ('general', 383),
     ('sequences', 383),
     ('lee', 383),
     ('points', 382),
     ('earlier', 382),
     ('gone', 379),
     ('check', 379),
     ('suspense', 378),
     ('recommended', 378),
     ('ten', 378),
     ('third', 377),
     ('business', 377),
     ('talk', 375),
     ('leaves', 375),
     ('beyond', 375),
     ('portrayal', 374),
     ('beautifully', 373),
     ('single', 372),
     ('bill', 372),
     ('plenty', 371),
     ('word', 371),
     ('whom', 370),
     ('falls', 370),
     ('scary', 369),
     ('non', 369),
     ('figure', 369),
     ('battle', 369),
     ('using', 368),
     ('return', 368),
     ('doubt', 367),
     ('add', 367),
     ('hear', 366),
     ('solid', 366),
     ('success', 366),
     ('jokes', 365),
     ('oh', 365),
     ('touching', 365),
     ('political', 365),
     ('hell', 364),
     ('awesome', 364),
     ('boys', 364),
     ('sexual', 362),
     ('recently', 362),
     ('dog', 362),
     ('please', 361),
     ('wouldn', 361),
     ('straight', 361),
     ('features', 361),
     ('forget', 360),
     ('setting', 360),
     ('lack', 360),
     ('married', 359),
     ('mark', 359),
     ('social', 357),
     ('interested', 356),
     ('adventure', 356),
     ('actual', 355),
     ('terrific', 355),
     ('sees', 355),
     ('brothers', 355),
     ('move', 354),
     ('call', 354),
     ('various', 353),
     ('theater', 353),
     ('dr', 353),
     ('animated', 352),
     ('western', 351),
     ('baby', 350),
     ('space', 350),
     ('leading', 348),
     ('disappointed', 348),
     ('portrayed', 346),
     ('aren', 346),
     ('screenplay', 345),
     ('smith', 345),
     ('towards', 344),
     ('hate', 344),
     ('noir', 343),
     ('outstanding', 342),
     ('decent', 342),
     ('kelly', 342),
     ('directors', 341),
     ('journey', 341),
     ('none', 340),
     ('looked', 340),
     ('effective', 340),
     ('storyline', 339),
     ('caught', 339),
     ('sci', 339),
     ('fi', 339),
     ('cold', 339),
     ('mary', 339),
     ('rich', 338),
     ('charming', 338),
     ('popular', 337),
     ('rare', 337),
     ('manages', 337),
     ('harry', 337),
     ('spirit', 336),
     ('appreciate', 335),
     ('open', 335),
     ('moves', 334),
     ('basically', 334),
     ('acted', 334),
     ('inside', 333),
     ('boring', 333),
     ('century', 333),
     ('mention', 333),
     ('deserves', 333),
     ('subtle', 333),
     ('pace', 333),
     ('familiar', 332),
     ('background', 332),
     ('ben', 331),
     ('creepy', 330),
     ('supposed', 330),
     ('secret', 329),
     ('die', 328),
     ('jim', 328),
     ('question', 327),
     ('effect', 327),
     ('natural', 327),
     ('impressive', 326),
     ('rate', 326),
     ('language', 326),
     ('saying', 325),
     ('intelligent', 325),
     ('telling', 324),
     ('realize', 324),
     ('material', 324),
     ('scott', 324),
     ('singing', 323),
     ('dancing', 322),
     ('visual', 321),
     ('adult', 321),
     ('imagine', 321),
     ('kept', 320),
     ('office', 320),
     ('uses', 319),
     ('pure', 318),
     ('wait', 318),
     ('stunning', 318),
     ('review', 317),
     ('previous', 317),
     ('copy', 317),
     ('seriously', 317),
     ('reading', 316),
     ('create', 316),
     ('hot', 316),
     ('created', 316),
     ('magic', 316),
     ('somehow', 316),
     ('stay', 315),
     ('attempt', 315),
     ('escape', 315),
     ('crazy', 315),
     ('air', 315),
     ('frank', 315),
     ('hands', 314),
     ('filled', 313),
     ('expected', 312),
     ('average', 312),
     ('surprisingly', 312),
     ('complex', 311),
     ('quickly', 310),
     ('successful', 310),
     ('studio', 310),
     ('plus', 309),
     ('male', 309),
     ('co', 307),
     ('images', 306),
     ('casting', 306),
     ('following', 306),
     ('minute', 306),
     ('exciting', 306),
     ('members', 305),
     ('follows', 305),
     ('themes', 305),
     ('german', 305),
     ('reasons', 305),
     ('e', 305),
     ('touch', 304),
     ('edge', 304),
     ('free', 304),
     ('cute', 304),
     ('genius', 304),
     ('outside', 303),
     ('reviews', 302),
     ('admit', 302),
     ('ok', 302),
     ('younger', 302),
     ('fighting', 301),
     ('odd', 301),
     ('master', 301),
     ('recent', 300),
     ('thanks', 300),
     ('break', 300),
     ('comment', 300),
     ('apart', 299),
     ('emotions', 298),
     ('lovely', 298),
     ('begin', 298),
     ('doctor', 297),
     ('party', 297),
     ('italian', 297),
     ('la', 296),
     ('missed', 296),
     ...]




```python
pos_neg_ratios = Counter()

for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio

for word,ratio in pos_neg_ratios.most_common():
    if(ratio > 1):
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))
```


```python
# words most frequently seen in a review with a "POSITIVE" label
pos_neg_ratios.most_common()
```




    [('edie', 4.6913478822291435),
     ('paulie', 4.0775374439057197),
     ('felix', 3.1527360223636558),
     ('polanski', 2.8233610476132043),
     ('matthau', 2.8067217286092401),
     ('victoria', 2.6810215287142909),
     ('mildred', 2.6026896854443837),
     ('gandhi', 2.5389738710582761),
     ('flawless', 2.451005098112319),
     ('superbly', 2.2600254785752498),
     ('perfection', 2.1594842493533721),
     ('astaire', 2.1400661634962708),
     ('captures', 2.0386195471595809),
     ('voight', 2.0301704926730531),
     ('wonderfully', 2.0218960560332353),
     ('powell', 1.9783454248084671),
     ('brosnan', 1.9547990964725592),
     ('lily', 1.9203768470501485),
     ('bakshi', 1.9029851043382795),
     ('lincoln', 1.9014583864844796),
     ('refreshing', 1.8551812956655511),
     ('breathtaking', 1.8481124057791867),
     ('bourne', 1.8478489358790986),
     ('lemmon', 1.8458266904983307),
     ('delightful', 1.8002701588959635),
     ('flynn', 1.7996646487351682),
     ('andrews', 1.7764919970972666),
     ('homer', 1.7692866133759964),
     ('beautifully', 1.7626953362841438),
     ('soccer', 1.7578579175523736),
     ('elvira', 1.7397031072720019),
     ('underrated', 1.7197859696029656),
     ('gripping', 1.7165360479904674),
     ('superb', 1.7091514458966952),
     ('delight', 1.6714733033535532),
     ('welles', 1.6677068205580761),
     ('sadness', 1.663505133704376),
     ('sinatra', 1.6389967146756448),
     ('touching', 1.637217476541176),
     ('timeless', 1.62924053973028),
     ('macy', 1.6211339521972916),
     ('unforgettable', 1.6177367152487956),
     ('favorites', 1.6158688027643908),
     ('stewart', 1.6119987332957739),
     ('sullivan', 1.6094379124341003),
     ('extraordinary', 1.6094379124341003),
     ('hartley', 1.6094379124341003),
     ('brilliantly', 1.5950491749820008),
     ('friendship', 1.5677652160335325),
     ('wonderful', 1.5645425925262093),
     ('palma', 1.5553706911638245),
     ('magnificent', 1.54663701119507),
     ('finest', 1.5462590108125689),
     ('jackie', 1.5439233053234738),
     ('ritter', 1.5404450409471491),
     ('tremendous', 1.5184661342283736),
     ('freedom', 1.5091151908062312),
     ('fantastic', 1.5048433868558566),
     ('terrific', 1.5026699370083942),
     ('noir', 1.493925025312256),
     ('sidney', 1.493925025312256),
     ('outstanding', 1.4910053152089213),
     ('pleasantly', 1.4894785973551214),
     ('mann', 1.4894785973551214),
     ('nancy', 1.488077055429833),
     ('marie', 1.4825711915553104),
     ('marvelous', 1.4739999415389962),
     ('excellent', 1.4647538505723599),
     ('ruth', 1.4596256342054401),
     ('stanwyck', 1.4412101187160054),
     ('widmark', 1.4350845252893227),
     ('splendid', 1.4271163556401458),
     ('chan', 1.423108334242607),
     ('exceptional', 1.4201959127955721),
     ('tender', 1.410986973710262),
     ('gentle', 1.4078005663408544),
     ('poignant', 1.4022947024663317),
     ('gem', 1.3932148039644643),
     ('amazing', 1.3919815802404802),
     ('chilling', 1.3862943611198906),
     ('fisher', 1.3862943611198906),
     ('davies', 1.3862943611198906),
     ('captivating', 1.3862943611198906),
     ('darker', 1.3652409519220583),
     ('april', 1.3499267169490159),
     ('kelly', 1.3461743673304654),
     ('blake', 1.3418425985490567),
     ('overlooked', 1.329135947279942),
     ('ralph', 1.32818673031261),
     ('bette', 1.3156767939059373),
     ('hoffman', 1.3150668518315229),
     ('cole', 1.3121863889661687),
     ('shines', 1.3049487216659381),
     ('powerful', 1.2999662776313934),
     ('notch', 1.2950456896547455),
     ('remarkable', 1.2883688239495823),
     ('pitt', 1.286210902562908),
     ('winters', 1.2833463918674481),
     ('vivid', 1.2762934659055623),
     ('gritty', 1.2757524867200667),
     ('giallo', 1.2745029551317739),
     ('portrait', 1.2704625455947689),
     ('innocence', 1.2694300209805796),
     ('psychiatrist', 1.2685113254635072),
     ('favorite', 1.2668956297860055),
     ('ensemble', 1.2656663733312759),
     ('stunning', 1.2622417124499117),
     ('burns', 1.259880436264232),
     ('garbo', 1.258954938743289),
     ('barbara', 1.2580400255962119),
     ('philip', 1.2527629684953681),
     ('panic', 1.2527629684953681),
     ('holly', 1.2527629684953681),
     ('carol', 1.2481440226390734),
     ('perfect', 1.246742480713785),
     ('appreciated', 1.2462482874741743),
     ('favourite', 1.2411123512753928),
     ('journey', 1.2367626271489269),
     ('rural', 1.235471471385307),
     ('bond', 1.2321436812926323),
     ('builds', 1.2305398317106577),
     ('brilliant', 1.2287554137664785),
     ('brooklyn', 1.2286654169163074),
     ('von', 1.225175011976539),
     ('recommended', 1.2163953243244932),
     ('unfolds', 1.2163953243244932),
     ('daniel', 1.20215296760895),
     ('perfectly', 1.1971931173405572),
     ('crafted', 1.1962507582320256),
     ('prince', 1.1939224684724346),
     ('troubled', 1.192138346678933),
     ('consequences', 1.1865810616140668),
     ('haunting', 1.1814999484738773),
     ('cinderella', 1.180052620608284),
     ('alexander', 1.1759989522835299),
     ('emotions', 1.1753049094563641),
     ('boxing', 1.1735135968412274),
     ('subtle', 1.1734135017508081),
     ('curtis', 1.1649873576129823),
     ('rare', 1.1566438362402944),
     ('loved', 1.1563661500586044),
     ('daughters', 1.1526795099383853),
     ('courage', 1.1438688802562305),
     ('dentist', 1.1426722784621401),
     ('highly', 1.1420208631618658),
     ('nominated', 1.1409146683587992),
     ('tony', 1.1397491942285991),
     ('draws', 1.1325138403437911),
     ('everyday', 1.1306150197542835),
     ('contrast', 1.1284652518177909),
     ('cried', 1.1213405397456659),
     ('fabulous', 1.1210851445201684),
     ('ned', 1.120591195386885),
     ('fay', 1.120591195386885),
     ('emma', 1.1184149159642893),
     ('sensitive', 1.113318436057805),
     ('smooth', 1.1089750757036563),
     ('dramas', 1.1080910326226534),
     ('today', 1.1050431789984001),
     ('helps', 1.1023091505494358),
     ('inspiring', 1.0986122886681098),
     ('jimmy', 1.0937696641923216),
     ('awesome', 1.0931328229034842),
     ('unique', 1.0881409888008142),
     ('tragic', 1.0871835928444868),
     ('intense', 1.0870514662670339),
     ('stellar', 1.0857088838322018),
     ('rival', 1.0822184788924332),
     ('provides', 1.0797081340289569),
     ('depression', 1.0782034170369026),
     ('shy', 1.0775588794702773),
     ('carrie', 1.076139432816051),
     ('blend', 1.0753554265038423),
     ('hank', 1.0736109864626924),
     ('diana', 1.0726368022648489),
     ('adorable', 1.0726368022648489),
     ('unexpected', 1.0722255334949147),
     ('achievement', 1.0668635903535293),
     ('bettie', 1.0663514264498881),
     ('happiness', 1.0632729222228008),
     ('glorious', 1.0608719606852626),
     ('davis', 1.0541605260972757),
     ('terrifying', 1.0525211814678428),
     ('beauty', 1.050410186850232),
     ('ideal', 1.0479685558493548),
     ('fears', 1.0467872208035236),
     ('hong', 1.0438040521731147),
     ('seasons', 1.0433496099930604),
     ('fascinating', 1.0414538748281612),
     ('carries', 1.0345904299031787),
     ('satisfying', 1.0321225473992768),
     ('definite', 1.0319209141694374),
     ('touched', 1.0296194171811581),
     ('greatest', 1.0248947127715422),
     ('creates', 1.0241097613701886),
     ('aunt', 1.023388867430522),
     ('walter', 1.022328983918479),
     ('spectacular', 1.0198314108149955),
     ('portrayal', 1.0189810189761024),
     ('ann', 1.0127808528183286),
     ('enterprise', 1.0116009116784799),
     ('musicals', 1.0096648026516135),
     ('deeply', 1.0094845087721023),
     ('incredible', 1.0061677561461084),
     ('mature', 1.0060195018402847),
     ('triumph', 0.99682959435816731),
     ('margaret', 0.99682959435816731),
     ('navy', 0.99493385919326827),
     ('harry', 0.99176919305006062),
     ('lucas', 0.990398704027877),
     ('sweet', 0.98966110487955483),
     ('joey', 0.98794672078059009),
     ('oscar', 0.98721905111049713),
     ('balance', 0.98649499054740353),
     ('warm', 0.98485340331145166),
     ('ages', 0.98449898190068863),
     ('guilt', 0.98082925301172619),
     ('glover', 0.98082925301172619),
     ('carrey', 0.98082925301172619),
     ('learns', 0.97881108885548895),
     ('unusual', 0.97788374278196932),
     ('sons', 0.97777581552483595),
     ('complex', 0.97761897738147796),
     ('essence', 0.97753435711487369),
     ('brazil', 0.9769153536905899),
     ('widow', 0.97650959186720987),
     ('solid', 0.97537964824416146),
     ('beautiful', 0.97326301262841053),
     ('holmes', 0.97246100334120955),
     ('awe', 0.97186058302896583),
     ('vhs', 0.97116734209998934),
     ('eerie', 0.97116734209998934),
     ('lonely', 0.96873720724669754),
     ('grim', 0.96873720724669754),
     ('sport', 0.96825047080486615),
     ('debut', 0.96508089604358704),
     ('destiny', 0.96343751029985703),
     ('thrillers', 0.96281074750904794),
     ('tears', 0.95977584381389391),
     ('rose', 0.95664202739772253),
     ('feelings', 0.95551144502743635),
     ('ginger', 0.95551144502743635),
     ('winning', 0.95471810900804055),
     ('stanley', 0.95387344302319799),
     ('cox', 0.95343027882361187),
     ('paris', 0.95278479030472663),
     ('heart', 0.95238806924516806),
     ('hooked', 0.95155887071161305),
     ('comfortable', 0.94803943018873538),
     ('mgm', 0.94446160884085151),
     ('masterpiece', 0.94155039863339296),
     ('themes', 0.94118828349588235),
     ('danny', 0.93967118051821874),
     ('anime', 0.93378388932167222),
     ('perry', 0.93328830824272613),
     ('joy', 0.93301752567946861),
     ('lovable', 0.93081883243706487),
     ('mysteries', 0.92953595862417571),
     ('hal', 0.92953595862417571),
     ('louis', 0.92871325187271225),
     ('charming', 0.92520609553210742),
     ('urban', 0.92367083917177761),
     ('allows', 0.92183091224977043),
     ('impact', 0.91815814604895041),
     ('italy', 0.91629073187415511),
     ('gradually', 0.91629073187415511),
     ('lifestyle', 0.91629073187415511),
     ('spy', 0.91289514287301687),
     ('treat', 0.91193342650519937),
     ('subsequent', 0.91056005716517008),
     ('kennedy', 0.90981821736853763),
     ('loving', 0.90967549275543591),
     ('surprising', 0.90937028902958128),
     ('quiet', 0.90648673177753425),
     ('winter', 0.90624039602065365),
     ('reveals', 0.90490540964902977),
     ('raw', 0.90445627422715225),
     ('funniest', 0.90078654533818991),
     ('pleased', 0.89994159387262562),
     ('norman', 0.89994159387262562),
     ('thief', 0.89874642222324552),
     ('season', 0.89827222637147675),
     ('secrets', 0.89794159320595857),
     ('colorful', 0.89705936994626756),
     ('highest', 0.8967461358011849),
     ('compelling', 0.89462923509297576),
     ('danes', 0.89248008318043659),
     ('castle', 0.88967708335606499),
     ('kudos', 0.88889175768604067),
     ('great', 0.88810470901464589),
     ('baseball', 0.88730319500090271),
     ('subtitles', 0.88730319500090271),
     ('bleak', 0.88730319500090271),
     ('winner', 0.88643776872447388),
     ('tragedy', 0.88563699078315261),
     ('todd', 0.88551907320740142),
     ('nicely', 0.87924946019380601),
     ('arthur', 0.87546873735389985),
     ('essential', 0.87373111745535925),
     ('gorgeous', 0.8731725250935497),
     ('fonda', 0.87294029100054127),
     ('eastwood', 0.87139541196626402),
     ('focuses', 0.87082835779739776),
     ('enjoyed', 0.87070195951624607),
     ('natural', 0.86997924506912838),
     ('intensity', 0.86835126958503595),
     ('witty', 0.86824103423244681),
     ('rob', 0.8642954367557748),
     ('worlds', 0.86377269759070874),
     ('health', 0.86113891179907498),
     ('magical', 0.85953791528170564),
     ('deeper', 0.85802182375017932),
     ('lucy', 0.85618680780444956),
     ('moving', 0.85566611005772031),
     ('lovely', 0.85290640004681306),
     ('purple', 0.8513711857748395),
     ('memorable', 0.84801189112086062),
     ('sings', 0.84729786038720367),
     ('craig', 0.84342938360928321),
     ('modesty', 0.84342938360928321),
     ('relate', 0.84326559685926517),
     ('episodes', 0.84223712084137292),
     ('strong', 0.84167135777060931),
     ('smith', 0.83959811108590054),
     ('tear', 0.83704136022001441),
     ('apartment', 0.83333115290549531),
     ('princess', 0.83290912293510388),
     ('disagree', 0.83290912293510388),
     ('kung', 0.83173334384609199),
     ('adventure', 0.83150561393278388),
     ('columbo', 0.82667857318446791),
     ('jake', 0.82667857318446791),
     ('adds', 0.82485652591452319),
     ('hart', 0.82472353834866463),
     ('strength', 0.82417544296634937),
     ('realizes', 0.82360006895738058),
     ('dave', 0.8232003088081431),
     ('childhood', 0.82208086393583857),
     ('forbidden', 0.81989888619908913),
     ('tight', 0.81883539572344199),
     ('surreal', 0.8178506590609026),
     ('manager', 0.81770990320170756),
     ('dancer', 0.81574950265227764),
     ('studios', 0.81093021621632877),
     ('con', 0.81093021621632877),
     ('miike', 0.80821651034473263),
     ('realistic', 0.80807714723392232),
     ('explicit', 0.80792269515237358),
     ('kurt', 0.8060875917405409),
     ('traditional', 0.80535917116687328),
     ('deals', 0.80535917116687328),
     ('holds', 0.80493858654806194),
     ('carl', 0.80437281567016972),
     ('touches', 0.80396154690023547),
     ('gene', 0.80314807577427383),
     ('albert', 0.8027669055771679),
     ('abc', 0.80234647252493729),
     ('cry', 0.80011930011211307),
     ('sides', 0.7995275841185171),
     ('develops', 0.79850769621777162),
     ('eyre', 0.79850769621777162),
     ('dances', 0.79694397424158891),
     ('oscars', 0.79633141679517616),
     ('legendary', 0.79600456599965308),
     ('hearted', 0.79492987486988764),
     ('importance', 0.79492987486988764),
     ('portraying', 0.79356592830699269),
     ('impressed', 0.79258107754813223),
     ('waters', 0.79112758892014912),
     ('empire', 0.79078565012386137),
     ('edge', 0.789774016249017),
     ('jean', 0.78845736036427028),
     ('environment', 0.78845736036427028),
     ('sentimental', 0.7864791203521645),
     ('captured', 0.78623760362595729),
     ('styles', 0.78592891401091158),
     ('daring', 0.78592891401091158),
     ('frank', 0.78275933924963248),
     ('tense', 0.78275933924963248),
     ('backgrounds', 0.78275933924963248),
     ('matches', 0.78275933924963248),
     ('gothic', 0.78209466657644144),
     ('sharp', 0.7814397877056235),
     ('achieved', 0.78015855754957497),
     ('court', 0.77947526404844247),
     ('steals', 0.7789140023173704),
     ('rules', 0.77844476107184035),
     ('colors', 0.77684619943659217),
     ('reunion', 0.77318988823348167),
     ('covers', 0.77139937745969345),
     ('tale', 0.77010822169607374),
     ('rain', 0.7683706017975328),
     ('denzel', 0.76804848873306297),
     ('stays', 0.76787072675588186),
     ('blob', 0.76725515271366718),
     ('maria', 0.76214005204689672),
     ('conventional', 0.76214005204689672),
     ('fresh', 0.76158434211317383),
     ('midnight', 0.76096977689870637),
     ('landscape', 0.75852993982279704),
     ('animated', 0.75768570169751648),
     ('titanic', 0.75666058628227129),
     ('sunday', 0.75666058628227129),
     ('spring', 0.7537718023763802),
     ('cagney', 0.7537718023763802),
     ('enjoyable', 0.75246375771636476),
     ('immensely', 0.75198768058287868),
     ('sir', 0.7507762933965817),
     ('nevertheless', 0.75067102469813185),
     ('driven', 0.74994477895307854),
     ('performances', 0.74883252516063137),
     ('memories', 0.74721440183022114),
     ('nowadays', 0.74721440183022114),
     ('simple', 0.74641420974143258),
     ('golden', 0.74533293373051557),
     ('leslie', 0.74533293373051557),
     ('lovers', 0.74497224842453125),
     ('relationship', 0.74484232345601786),
     ('supporting', 0.74357803418683721),
     ('che', 0.74262723782331497),
     ('packed', 0.7410032017375805),
     ('trek', 0.74021469141793106),
     ('provoking', 0.73840377214806618),
     ('strikes', 0.73759894313077912),
     ('depiction', 0.73682224406260699),
     ('emotional', 0.73678211645681524),
     ('secretary', 0.7366322924996842),
     ('influenced', 0.73511137965897755),
     ('florida', 0.73511137965897755),
     ('germany', 0.73288750920945944),
     ('brings', 0.73142936713096229),
     ('lewis', 0.73129894652432159),
     ('elderly', 0.73088750854279239),
     ('owner', 0.72743625403857748),
     ('streets', 0.72666987259858895),
     ('henry', 0.72642196944481741),
     ('portrays', 0.72593700338293632),
     ('bears', 0.7252354951114458),
     ('china', 0.72489587887452556),
     ('anger', 0.72439972406404984),
     ('society', 0.72433010799663333),
     ('available', 0.72415741730250549),
     ('best', 0.72347034060446314),
     ('bugs', 0.72270598280148979),
     ('magic', 0.71878961117328299),
     ('delivers', 0.71846498854423513),
     ('verhoeven', 0.71846498854423513),
     ('jim', 0.71783979315031676),
     ('donald', 0.71667767797013937),
     ('endearing', 0.71465338578090898),
     ('relationships', 0.71393795022901896),
     ('greatly', 0.71256526641704687),
     ('charlie', 0.71024161391924534),
     ('brad', 0.71024161391924534),
     ('simon', 0.70967648251115578),
     ('effectively', 0.70914752190638641),
     ('march', 0.70774597998109789),
     ('atmosphere', 0.70744773070214162),
     ('influence', 0.70733181555190172),
     ('genius', 0.706392407309966),
     ('emotionally', 0.70556970055850243),
     ('ken', 0.70526854109229009),
     ('identity', 0.70484322032313651),
     ('sophisticated', 0.70470800296102132),
     ('dan', 0.70457587638356811),
     ('andrew', 0.70329955202396321),
     ('india', 0.70144598337464037),
     ('roy', 0.69970458110610434),
     ('surprisingly', 0.6995780708902356),
     ('sky', 0.69780919366575667),
     ('romantic', 0.69664981111114743),
     ('match', 0.69566924999265523),
     ('meets', 0.69314718055994529),
     ('cowboy', 0.69314718055994529),
     ('wave', 0.69314718055994529),
     ('bitter', 0.69314718055994529),
     ('patient', 0.69314718055994529),
     ('stylish', 0.69314718055994529),
     ('britain', 0.69314718055994529),
     ('affected', 0.69314718055994529),
     ('beatty', 0.69314718055994529),
     ('love', 0.69198533541937324),
     ('paul', 0.68980827929443067),
     ('andy', 0.68846333124751902),
     ('performance', 0.68797386327972465),
     ('patrick', 0.68645819240914863),
     ('unlike', 0.68546468438792907),
     ('brooks', 0.68433655087779044),
     ('refuses', 0.68348526964820844),
     ('award', 0.6824518914431974),
     ('complaint', 0.6824518914431974),
     ('ride', 0.68229716453587952),
     ('dawson', 0.68171848473632257),
     ('luke', 0.68158635815886937),
     ('wells', 0.68087708796813096),
     ('france', 0.6804081547825156),
     ('sports', 0.68007509899259255),
     ('handsome', 0.68007509899259255),
     ('directs', 0.67875844310784572),
     ('rebel', 0.67875844310784572),
     ('greater', 0.67605274720064523),
     ('dreams', 0.67599410133369586),
     ('effective', 0.67565402311242806),
     ('interpretation', 0.67479804189174875),
     ('works', 0.67445504754779284),
     ('brando', 0.67445504754779284),
     ('noble', 0.6737290947028437),
     ('paced', 0.67314651385327573),
     ('le', 0.67067432470788668),
     ('master', 0.67015766233524654),
     ('h', 0.6696166831497512),
     ('rings', 0.66904962898088483),
     ('easy', 0.66895995494594152),
     ('city', 0.66820823221269321),
     ('sunshine', 0.66782937257565544),
     ('succeeds', 0.66647893347778397),
     ('relations', 0.664159643686693),
     ('england', 0.66387679825983203),
     ('glimpse', 0.66329421741026418),
     ('aired', 0.66268797307523675),
     ('sees', 0.66263163663399482),
     ('both', 0.66248336767382998),
     ('definitely', 0.66199789483898808),
     ('imaginative', 0.66139848224536502),
     ('appreciate', 0.66083893732728749),
     ('tricks', 0.66071190480679143),
     ('striking', 0.66071190480679143),
     ('carefully', 0.65999497324304479),
     ('complicated', 0.65981076029235353),
     ('perspective', 0.65962448852130173),
     ('trilogy', 0.65877953705573755),
     ('future', 0.65834665141052828),
     ('lion', 0.65742909795786608),
     ('douglas', 0.65540685257709819),
     ('victor', 0.65540685257709819),
     ('inspired', 0.65459851044271034),
     ('marriage', 0.65392646740666405),
     ('demands', 0.65392646740666405),
     ('father', 0.65172321672194655),
     ('page', 0.65123628494430852),
     ('instant', 0.65058756614114943),
     ('era', 0.6495567444850836),
     ('ruthless', 0.64934455790155243),
     ('saga', 0.64934455790155243),
     ('joan', 0.64891392558311978),
     ('joseph', 0.64841128671855386),
     ('workers', 0.64829661439459352),
     ('fantasy', 0.64726757480925168),
     ('distant', 0.64551913157069074),
     ('accomplished', 0.64551913157069074),
     ('manhattan', 0.64435701639051324),
     ('personal', 0.64355023942057321),
     ('meeting', 0.64313675998528386),
     ('individual', 0.64313675998528386),
     ('pushing', 0.64313675998528386),
     ('pleasant', 0.64250344774119039),
     ('brave', 0.64185388617239469),
     ('william', 0.64083139119578469),
     ('hudson', 0.64077919504262937),
     ('friendly', 0.63949446706762514),
     ('eccentric', 0.63907995928966954),
     ('awards', 0.63875310849414646),
     ('jack', 0.63838309514997038),
     ('seeking', 0.63808740337691783),
     ('divorce', 0.63757732940513456),
     ('colonel', 0.63757732940513456),
     ('jane', 0.63443957973316734),
     ('keeping', 0.63414883979798953),
     ('gives', 0.63383568159497883),
     ('ted', 0.63342794585832296),
     ('animation', 0.63208692379869902),
     ('progress', 0.6317782341836532),
     ('larger', 0.63127177684185776),
     ('concert', 0.63127177684185776),
     ('nation', 0.6296337748376194),
     ('albeit', 0.62739580299716491),
     ('adapted', 0.62613647027698516),
     ('discovers', 0.62542900650499444),
     ('classic', 0.62504956428050518),
     ('segment', 0.62335141862440335),
     ('morgan', 0.62303761437291871),
     ('mouse', 0.62294292188669675),
     ('impressive', 0.62211140744319349),
     ('artist', 0.62168821657780038),
     ('ultimate', 0.62168821657780038),
     ('griffith', 0.62117368093485603),
     ('drew', 0.62082651898031915),
     ('emily', 0.62082651898031915),
     ('moved', 0.6197197120051281),
     ('families', 0.61903920840622351),
     ('profound', 0.61903920840622351),
     ('innocent', 0.61851219917136446),
     ('versions', 0.61730910416844087),
     ('eddie', 0.61691981517206107),
     ('criticism', 0.61651395453902935),
     ('nature', 0.61594514653194088),
     ('recognized', 0.61518563909023349),
     ('sexuality', 0.61467556511845012),
     ('contract', 0.61400986000122149),
     ('brian', 0.61344043794920278),
     ('remembered', 0.6131044728864089),
     ('determined', 0.6123858239154869),
     ('offers', 0.61207935747116349),
     ('pleasure', 0.61195702582993206),
     ('washington', 0.61180154110599294),
     ('images', 0.61159731359583758),
     ('games', 0.61067095873570676),
     ('academy', 0.60872983874736208),
     ('fashioned', 0.60798937221963845),
     ('melodrama', 0.60749173598145145),
     ('rough', 0.60613580357031549),
     ('charismatic', 0.60613580357031549),
     ('peoples', 0.60613580357031549),
     ('dealing', 0.60517840761398811),
     ('fine', 0.60496962268013299),
     ('tap', 0.60391604683200273),
     ('trio', 0.60157998703445481),
     ('russell', 0.60120968523425966),
     ('figures', 0.60077386042893011),
     ('ward', 0.60005675749393339),
     ('shine', 0.59911823091166894),
     ('brady', 0.59911823091166894),
     ('job', 0.59845562125168661),
     ('satisfied', 0.59652034487087369),
     ('river', 0.59637962862495086),
     ('brown', 0.595773016534769),
     ('believable', 0.59566072133302495),
     ('always', 0.59470710774669278),
     ('bound', 0.59470710774669278),
     ('hall', 0.5933967777928858),
     ('cook', 0.5916777203950857),
     ('claire', 0.59136448625000293),
     ('broadway', 0.59033768669372433),
     ('anna', 0.58778666490211906),
     ('peace', 0.58628403501758408),
     ('visually', 0.58539431926349916),
     ('morality', 0.58525821854876026),
     ('falk', 0.58525821854876026),
     ('growing', 0.58466653756587539),
     ('experiences', 0.58314628534561685),
     ('stood', 0.58314628534561685),
     ('touch', 0.58122926435596001),
     ('lives', 0.5810976767513224),
     ('kubrick', 0.58066919713325493),
     ('timing', 0.58047401805583243),
     ('expressions', 0.57981849525294216),
     ('struggles', 0.57981849525294216),
     ('authentic', 0.57848427223980559),
     ('helen', 0.57763429343810091),
     ('pre', 0.57700753064729182),
     ('quirky', 0.5753641449035618),
     ('young', 0.57531672344534313),
     ('inner', 0.57454143815209846),
     ('mexico', 0.57443087372056334),
     ('clint', 0.57380042292737909),
     ('sisters', 0.57286101468544337),
     ('realism', 0.57226528899949558),
     ('french', 0.5720692490067093),
     ('personalities', 0.5720692490067093),
     ('surprises', 0.57113222999698177),
     ('adventures', 0.57113222999698177),
     ('overcome', 0.5697681593994407),
     ('timothy', 0.56953322459276867),
     ('tales', 0.56909453188996639),
     ('war', 0.56843317302781682),
     ('civil', 0.5679840376059393),
     ('countries', 0.56737779327091187),
     ('streep', 0.56710645966458029),
     ('tradition', 0.56685345523565323),
     ('oliver', 0.56673325570428668),
     ('australia', 0.56580775818334383),
     ('understanding', 0.56531380905006046),
     ('players', 0.56509525370004821),
     ('knowing', 0.56489284503626647),
     ('rogers', 0.56421349718405212),
     ('suspenseful', 0.56368911332305849),
     ('variety', 0.56368911332305849),
     ('true', 0.56281525180810066),
     ('jr', 0.56220982311246936),
     ('psychological', 0.56108745854687891),
     ('sent', 0.55961578793542266),
     ('grand', 0.55961578793542266),
     ('branagh', 0.55961578793542266),
     ('reminiscent', 0.55961578793542266),
     ('performing', 0.55961578793542266),
     ('wealth', 0.55961578793542266),
     ('overwhelming', 0.55961578793542266),
     ('odds', 0.55961578793542266),
     ('brothers', 0.55891181043362848),
     ('howard', 0.55811089675600245),
     ('david', 0.55693122256475369),
     ('generation', 0.55628799784274796),
     ('grow', 0.55612538299565417),
     ('survival', 0.55594605904646033),
     ('mainstream', 0.55574731115750231),
     ('dick', 0.55431073570572953),
     ('charm', 0.55288175575407861),
     ('kirk', 0.55278982286502287),
     ('twists', 0.55244729845681018),
     ('gangster', 0.55206858230003986),
     ('jeff', 0.55179306225421365),
     ('family', 0.55116244510065526),
     ('tend', 0.55053307336110335),
     ('thanks', 0.55049088015842218),
     ('world', 0.54744234723432639),
     ('sutherland', 0.54743536937855164),
     ('life', 0.54695514434959924),
     ('disc', 0.54654370636806993),
     ('bug', 0.54654370636806993),
     ('tribute', 0.5455111817538808),
     ('europe', 0.54522705048332309),
     ('sacrifice', 0.54430155296238014),
     ('color', 0.54405127139431109),
     ('superior', 0.54333490233128523),
     ('york', 0.54318235866536513),
     ('pulls', 0.54266622962164945),
     ('jackson', 0.54232429082536171),
     ('hearts', 0.54232429082536171),
     ('enjoy', 0.54124285135906114),
     ('redemption', 0.54056759296472823),
     ('madness', 0.540384426007535),
     ('stands', 0.5389965007326869),
     ('trial', 0.5389965007326869),
     ('greek', 0.5389965007326869),
     ('hamilton', 0.5389965007326869),
     ('each', 0.5388212312554177),
     ('faithful', 0.53773307668591508),
     ('received', 0.5372768098531604),
     ('documentaries', 0.53714293208336406),
     ('jealous', 0.53714293208336406),
     ('different', 0.53709860682460819),
     ('describes', 0.53680111016925136),
     ('shorts', 0.53596159703753288),
     ('brilliance', 0.53551823635636209),
     ('mountains', 0.53492317534505118),
     ('share', 0.53408248593025787),
     ('dealt', 0.53408248593025787),
     ('providing', 0.53329847961804933),
     ('explore', 0.53329847961804933),
     ('series', 0.5325809226575603),
     ('fellow', 0.5323318289869543),
     ('loves', 0.53062825106217038),
     ('revolution', 0.53062825106217038),
     ('olivier', 0.53062825106217038),
     ('roman', 0.53062825106217038),
     ('century', 0.53002783074992665),
     ('musical', 0.52966871156747064),
     ('heroic', 0.52925932545482868),
     ('approach', 0.52806743020049673),
     ('ironically', 0.52806743020049673),
     ('temple', 0.52806743020049673),
     ('moves', 0.5279372642387119),
     ('gift', 0.52702030968597136),
     ('julie', 0.52609309589677911),
     ('tells', 0.52415107836314001),
     ('radio', 0.52394671172868779),
     ('uncle', 0.52354439617376536),
     ('union', 0.52324814376454787),
     ('deep', 0.52309571635780505),
     ('reminds', 0.52157841554225237),
     ('famous', 0.52118841080153722),
     ('jazz', 0.52053443789295151),
     ('dennis', 0.51987545928590861),
     ('epic', 0.51919387343650736),
     ('adult', 0.519167695083386),
     ('shows', 0.51915322220375304),
     ('performed', 0.5191244265806858),
     ('demons', 0.5191244265806858),
     ('discovered', 0.51879379341516751),
     ('eric', 0.51879379341516751),
     ('youth', 0.5185626062681431),
     ('human', 0.51851411224987087),
     ('tarzan', 0.51813827061227724),
     ('ourselves', 0.51794309153485463),
     ('wwii', 0.51758240622887042),
     ('passion', 0.5162164724008671),
     ('desire', 0.51607497965213445),
     ('pays', 0.51581316527702981),
     ('dirty', 0.51557622652458857),
     ('fox', 0.51557622652458857),
     ('sympathetic', 0.51546600332249293),
     ('symbolism', 0.51546600332249293),
     ('attitude', 0.51530993621331933),
     ('appearances', 0.51466440007315639),
     ('jeremy', 0.51466440007315639),
     ('fun', 0.51439068993048687),
     ('south', 0.51420972175023116),
     ('arrives', 0.51409894911095988),
     ('present', 0.51341965894303732),
     ('com', 0.51326167856387173),
     ('smile', 0.51265880484765169),
     ('alan', 0.51082562376599072),
     ('ring', 0.51082562376599072),
     ('visit', 0.51082562376599072),
     ('fits', 0.51082562376599072),
     ('provided', 0.51082562376599072),
     ('carter', 0.51082562376599072),
     ('aging', 0.51082562376599072),
     ('countryside', 0.51082562376599072),
     ('begins', 0.51015650363396647),
     ('success', 0.50900578704900468),
     ('japan', 0.50900578704900468),
     ('accurate', 0.50895471583017893),
     ('proud', 0.50800474742434931),
     ('daily', 0.5075946031845443),
     ('karloff', 0.50724780241810674),
     ('atmospheric', 0.50724780241810674),
     ('recently', 0.50714914903668207),
     ('fu', 0.50704490092608467),
     ('horrors', 0.50656122497953315),
     ('finding', 0.50637127341661037),
     ('lust', 0.5059356384717989),
     ('hitchcock', 0.50574947073413001),
     ('among', 0.50334004951332734),
     ('viewing', 0.50302139827440906),
     ('investigation', 0.50262885656181222),
     ('shining', 0.50262885656181222),
     ('duo', 0.5020919437972361),
     ('cameron', 0.5020919437972361),
     ('finds', 0.50128303100539795),
     ('contemporary', 0.50077528791248915),
     ('genuine', 0.50046283673044401),
     ('frightening', 0.49995595152908684),
     ('plays', 0.49975983848890226),
     ('age', 0.49941323171424595),
     ('position', 0.49899116611898781),
     ('continues', 0.49863035067217237),
     ('roles', 0.49839716550752178),
     ('james', 0.49837216269470402),
     ('individuals', 0.49824684155913052),
     ('brought', 0.49783842823917956),
     ('hilarious', 0.49714551986191058),
     ('brutal', 0.49681488669639234),
     ('appropriate', 0.49643688631389105),
     ('dance', 0.49581998314812048),
     ('league', 0.49578774640145024),
     ('helping', 0.49578774640145024),
     ('answers', 0.49578774640145024),
     ('stunts', 0.49561620510246196),
     ('traveling', 0.49532143723002542),
     ('thoroughly', 0.49414593456733524),
     ('depicted', 0.49317068852726992),
     ('combination', 0.49247648509779424),
     ('honor', 0.49247648509779424),
     ('differences', 0.49247648509779424),
     ('fully', 0.49213349075383811),
     ('tracy', 0.49159426183810306),
     ('battles', 0.49140753790888908),
     ('possibility', 0.49112055268665822),
     ('romance', 0.4901589869574316),
     ('initially', 0.49002249613622745),
     ('happy', 0.4898997500608791),
     ('crime', 0.48977221456815834),
     ('singing', 0.4893852925281213),
     ('especially', 0.48901267837860624),
     ('shakespeare', 0.48754793889664511),
     ('hugh', 0.48729512635579658),
     ('detail', 0.48609484250827351),
     ('julia', 0.48550781578170082),
     ('san', 0.48550781578170082),
     ('guide', 0.48550781578170082),
     ('desperation', 0.48550781578170082),
     ('companion', 0.48550781578170082),
     ('strongly', 0.48460242866688824),
     ('necessary', 0.48302334245403883),
     ('humanity', 0.48265474679929443),
     ('drama', 0.48221998493060503),
     ('nonetheless', 0.48183808689273838),
     ('intrigue', 0.48183808689273838),
     ('warming', 0.48183808689273838),
     ('cuba', 0.48183808689273838),
     ('planned', 0.47957308026188628),
     ('pictures', 0.47929937011921681),
     ('broadcast', 0.47849024312305422),
     ('nine', 0.47803580094299974),
     ('settings', 0.47743860773325364),
     ('history', 0.47732966933780852),
     ('ordinary', 0.47725880012690741),
     ('trade', 0.47692407209030935),
     ('official', 0.47608267532211779),
     ('primary', 0.47608267532211779),
     ('episode', 0.47529620261150429),
     ('role', 0.47520268270188676),
     ('spirit', 0.47477690799839323),
     ('grey', 0.47409361449726067),
     ('ways', 0.47323464982718205),
     ('cup', 0.47260441094579297),
     ('piano', 0.47260441094579297),
     ('familiar', 0.47241617565111949),
     ('sinister', 0.47198579044972683),
     ('reveal', 0.47171449364936496),
     ('max', 0.47150852042515579),
     ('dated', 0.47121648567094482),
     ('losing', 0.47000362924573563),
     ('discovery', 0.47000362924573563),
     ('vicious', 0.47000362924573563),
     ('genuinely', 0.46871413841586385),
     ('hatred', 0.46734051182625186),
     ('mistaken', 0.46702300110759781),
     ('dream', 0.46608972992459924),
     ('challenge', 0.46608972992459924),
     ('crisis', 0.46575733836428446),
     ('photographed', 0.46488852857896512),
     ('critics', 0.46430560813109778),
     ('bird', 0.46430560813109778),
     ('machines', 0.46430560813109778),
     ('born', 0.46411383518967209),
     ('detective', 0.4636633473511525),
     ('higher', 0.46328467899699055),
     ('remains', 0.46262352194811296),
     ('inevitable', 0.46262352194811296),
     ('soviet', 0.4618180446592961),
     ('ryan', 0.46134556650262099),
     ('african', 0.46112595521371813),
     ('smaller', 0.46081520319132935),
     ('techniques', 0.46052488529119184),
     ('information', 0.46034171833399862),
     ('deserved', 0.45999798712841444),
     ('lynch', 0.45953232937844013),
     ('spielberg', 0.45953232937844013),
     ('cynical', 0.45953232937844013),
     ('tour', 0.45953232937844013),
     ('francisco', 0.45953232937844013),
     ('struggle', 0.45911782160048453),
     ('language', 0.45902121257712653),
     ('visual', 0.45823514408822852),
     ('warner', 0.45724137763188427),
     ('social', 0.45720078250735313),
     ('reality', 0.45719346885019546),
     ('hidden', 0.45675840249571492),
     ('breaking', 0.45601738727099561),
     ('sometimes', 0.45563021171182794),
     ('modern', 0.45500247579345005),
     ('surfing', 0.45425527227759638),
     ('popular', 0.45410691533051023),
     ('surprised', 0.4534409399850382),
     ('follows', 0.45245361754408348),
     ('keeps', 0.45234869400701483),
     ('john', 0.4520909494482197),
     ('mixed', 0.45198512374305722),
     ('defeat', 0.45198512374305722),
     ('justice', 0.45142724367280018),
     ('treasure', 0.45083371313801535),
     ('presents', 0.44973793178615257),
     ('years', 0.44919197032104968),
     ('chief', 0.44895022004790319),
     ('shadows', 0.44802472252696035),
     ('closely', 0.44701411102103689),
     ('segments', 0.44701411102103689),
     ('lose', 0.44658335503763702),
     ('caine', 0.44628710262841953),
     ('caught', 0.44610275383999071),
     ('hamlet', 0.44558510189758965),
     ('chinese', 0.44507424620321018),
     ('welcome', 0.44438052435783792),
     ('birth', 0.44368632092836219),
     ('represents', 0.44320543609101143),
     ('puts', 0.44279106572085081),
     ('visuals', 0.44183275227903923),
     ('fame', 0.44183275227903923),
     ('closer', 0.44183275227903923),
     ('web', 0.44183275227903923),
     ('criminal', 0.4412745608048752),
     ('minor', 0.4409224199448939),
     ('jon', 0.44086703515908027),
     ('liked', 0.44074991514020723),
     ('restaurant', 0.44031183943833246),
     ('de', 0.43983275161237217),
     ('flaws', 0.43983275161237217),
     ('searching', 0.4393666597838457),
     ('rap', 0.43891304217570443),
     ('light', 0.43884433018199892),
     ('elizabeth', 0.43872232986464677),
     ('marry', 0.43861731542506488),
     ('learned', 0.43825493093115531),
     ('controversial', 0.43825493093115531),
     ('oz', 0.43825493093115531),
     ('slowly', 0.43785660389939979),
     ('comedic', 0.43721380642274466),
     ('wayne', 0.43721380642274466),
     ('thrilling', 0.43721380642274466),
     ('bridge', 0.43721380642274466),
     ('married', 0.43658501682196887),
     ('nazi', 0.4361020775700542),
     ('murder', 0.4353180712578455),
     ('physical', 0.4353180712578455),
     ('johnny', 0.43483971678806865),
     ('michelle', 0.43445264498141672),
     ('wallace', 0.43403848055222038),
     ('comedies', 0.43395706390247063),
     ('silent', 0.43395706390247063),
     ('played', 0.43387244114515305),
     ('international', 0.43363598507486073),
     ('vision', 0.43286408229627887),
     ('intelligent', 0.43196704885367099),
     ('shop', 0.43078291609245434),
     ('also', 0.43036720209769169),
     ('levels', 0.4302451371066513),
     ('miss', 0.43006426712153217),
     ('movement', 0.4295626596872249),
     ...]




```python
# words most frequently seen in a review with a "NEGATIVE" label
list(reversed(pos_neg_ratios.most_common()))[0:30]
```




    [('boll', -4.0778152602708904),
     ('uwe', -3.9218753018711578),
     ('seagal', -3.3202501058581921),
     ('unwatchable', -3.0269848170580955),
     ('stinker', -2.9876839403711624),
     ('mst', -2.7753833211707968),
     ('incoherent', -2.7641396677532537),
     ('unfunny', -2.5545257844967644),
     ('waste', -2.4907515123361046),
     ('blah', -2.4475792789485005),
     ('horrid', -2.3715779644809971),
     ('pointless', -2.3451073877136341),
     ('atrocious', -2.3187369339642556),
     ('redeeming', -2.2667790015910296),
     ('prom', -2.2601040980178784),
     ('drivel', -2.2476029585766928),
     ('lousy', -2.2118080125207054),
     ('worst', -2.1930856334332267),
     ('laughable', -2.172468615469592),
     ('awful', -2.1385076866397488),
     ('poorly', -2.1326133844207011),
     ('wasting', -2.1178155545614512),
     ('remotely', -2.111046881095167),
     ('existent', -2.0024805005437076),
     ('boredom', -1.9241486572738005),
     ('miserably', -1.9216610938019989),
     ('sucks', -1.9166645809588516),
     ('uninspired', -1.9131499212248517),
     ('lame', -1.9117232884159072),
     ('insult', -1.9085323769376259)]



# Transforming Text into Numbers


```python
from IPython.display import Image

review = "This was a horrible, terrible movie."

Image(filename='sentiment_network.png')
```




![png](/assets/img/neural_network/output_18_0.png)




```python
review = "The movie was excellent"

Image(filename='sentiment_network_pos.png')
```




![png](/assets/img/neural_network/output_19_0.png)



# Project 2: Creating the Input/Output Data


```python
vocab = set(total_counts.keys())
vocab_size = len(vocab)
print(vocab_size)
```

    74074



```python
list(vocab)
```




    ['',
     'werewoves',
     'endowments',
     'palace',
     'persiflage',
     'slasherville',
     'locally',
     'unrecycled',
     'spearhead',
     'allyson',
     'manhating',
     'bartok',
     'gretorexes',
     'soaks',
     'protestations',
     'superimposes',
     'theirry',
     'yaqui',
     'contrives',
     'accessorizing',
     'arg',
     'sanguine',
     'batouch',
     'asked',
     'animals',
     'cockpits',
     'gorilla',
     'diculous',
     'establishing',
     'kagemusha',
     'sketches',
     'rebuilt',
     'perniciously',
     'socioeconomic',
     'ladylike',
     'prognostication',
     'blech',
     'sugarbabe',
     'desk',
     'fez',
     'accents',
     'speach',
     'rooster',
     'effort',
     'bodega',
     'dong',
     'preordained',
     'dubliners',
     'vili',
     'imperatives',
     'artifices',
     'wieder',
     'climate',
     'whoopdedoodles',
     'quatermass',
     'inveterate',
     'memorandum',
     'crucially',
     'bulimics',
     'misdrawing',
     'plympton',
     'fireballs',
     'verdant',
     'testi',
     'undeservingly',
     'lusted',
     'shylock',
     'disinfecting',
     'boxer',
     'givney',
     'hs',
     'loser',
     'civics',
     'volcano',
     'jur',
     'mohnish',
     'candidates',
     'assemble',
     'simi',
     'resort',
     'hessling',
     'starbase',
     'orgolini',
     'starrett',
     'weaker',
     'transcending',
     'levitate',
     'spurns',
     'contradictory',
     'cambreau',
     'latvia',
     'kirkpatrick',
     'betty',
     'agnostic',
     'sosa',
     'kanji',
     'swill',
     'millinium',
     'macgregor',
     'brd',
     'ariete',
     'assassins',
     'disscusion',
     'legislative',
     'dwars',
     'controller',
     'hadass',
     'vega',
     'bends',
     'glock',
     'spacewalk',
     'va',
     'offa',
     'winfield',
     'somewhat',
     'yates',
     'vinyl',
     'complicity',
     'bela',
     'squishes',
     'rippings',
     'eyed',
     'amatuerish',
     'desilva',
     'christmass',
     'briley',
     'bakhtyari',
     'unmasked',
     'huffman',
     'fallacious',
     'problem',
     'sieger',
     'koma',
     'grovelling',
     'incl',
     'farlinger',
     'teasers',
     'huff',
     'untried',
     'crocker',
     'dansu',
     'scammers',
     'popsicle',
     'arthritic',
     'grubs',
     'exemplar',
     'racial',
     'verbiage',
     'saloshin',
     'painlessly',
     'harewood',
     'shart',
     'keepers',
     'archrivals',
     'longish',
     'batmobile',
     'shakespearian',
     'bestselling',
     'spewing',
     'midlands',
     'trattoria',
     'greenaway',
     'gestapo',
     'ed',
     'huns',
     'bloch',
     'mashall',
     'versy',
     'david',
     'sicilian',
     'propositioned',
     'eighty',
     'carridine',
     'delicates',
     'veering',
     'columbus',
     'dunning',
     'mercantile',
     'rape',
     'purely',
     'rediscovered',
     'abstinence',
     'clunes',
     'emerson',
     'judgments',
     'lawful',
     'celebration',
     'affirmative',
     'sedately',
     'sng',
     'inuindo',
     'mosely',
     'bungalow',
     'ninga',
     'dripped',
     'itallian',
     'himalaya',
     'shikoku',
     'braik',
     'grousing',
     'nair',
     'forrester',
     'elemental',
     'allegations',
     'delilah',
     'boneheaded',
     'baltimoreans',
     'dunebuggies',
     'taguchi',
     'coleseum',
     'saratoga',
     'ninotchka',
     'afganistan',
     'genorisity',
     'haff',
     'jennilee',
     'jesues',
     'dwarfs',
     'enchilada',
     'feminist',
     'ghettoisation',
     'handlebar',
     'antagonistic',
     'marian',
     'crichton',
     'ryo',
     'mean',
     'inheritance',
     'presently',
     'pear',
     'inequality',
     'stately',
     'nooo',
     'obscurities',
     'determinedly',
     'solemn',
     'sullenly',
     'machism',
     'tingled',
     'maschera',
     'tristran',
     'mendoza',
     'baked',
     'jonatha',
     'lowly',
     'halliwell',
     'msted',
     'rodann',
     'hunkered',
     'cashmere',
     'chevalia',
     'jakub',
     'dobermann',
     'overexxagerating',
     'pfennig',
     'fisted',
     'mcelwee',
     'chief',
     'parlor',
     'browbeating',
     'parasol',
     'negligible',
     'kira',
     'monceau',
     'blew',
     'odete',
     'muco',
     'predominantly',
     'levon',
     'discourage',
     'fragmented',
     'vandermey',
     'etude',
     'mitch',
     'sandbag',
     'bending',
     'dizzying',
     'mover',
     'rewired',
     'awww',
     'di',
     'bejesus',
     'wallet',
     'uprooting',
     'atari',
     'dreamlike',
     'exacted',
     'harbouring',
     'indiscreet',
     'turks',
     'gems',
     'hoboken',
     'yalom',
     'rooftop',
     'howit',
     'tolson',
     'tulane',
     'reductive',
     'catharthic',
     'famarialy',
     'sista',
     'ghidorah',
     'ngoombujarra',
     'intently',
     'jlu',
     'kyrptonite',
     'hilda',
     'census',
     'baguette',
     'mondrians',
     'advisable',
     'mcnee',
     'candlelit',
     'affability',
     'intercut',
     'installations',
     'elliptical',
     'washer',
     'colt',
     'pevensie',
     'outshined',
     'despotic',
     'suares',
     'privates',
     'scrabble',
     'milliardo',
     'booting',
     'rowan',
     'golmaal',
     'pueblos',
     'msf',
     'giulietta',
     'phili',
     'pleaseee',
     'connecticute',
     'rosnelski',
     'tenebra',
     'bako',
     'blessings',
     'smudge',
     'cya',
     'pummel',
     'brocks',
     'homere',
     'propellant',
     'deliveried',
     'finisham',
     'newsradio',
     'bernie',
     'gouden',
     'enchant',
     'bessie',
     'semisubmerged',
     'extraterrestrial',
     'believably',
     'accomplice',
     'dooku',
     'baja',
     'met',
     'circulate',
     'disobeyed',
     'quakerly',
     'overstyling',
     'softens',
     'units',
     'shaye',
     'starters',
     'gripes',
     'nightmarish',
     'patriotic',
     'goodtimes',
     'stroheim',
     'debit',
     'prissies',
     'woebegone',
     'deputies',
     'awkwardness',
     'obama',
     'tarazu',
     'kendra',
     'patriots',
     'helpfuls',
     'mightily',
     'polemical',
     'unruly',
     'planing',
     'paperhouse',
     'sororities',
     'pym',
     'therin',
     'tarkovsky',
     'rdiger',
     'resembling',
     'gimmicks',
     'iler',
     'lineal',
     'taming',
     'mortenson',
     'waugh',
     'furies',
     'grufford',
     'hammill',
     'plunkett',
     'paterson',
     'konishita',
     'immorality',
     'angelos',
     'kebbel',
     'tamiroff',
     'boen',
     'rivalry',
     'ethnic',
     'funner',
     'troops',
     'deadeningly',
     'watcha',
     'dhiraj',
     'haranguing',
     'dejas',
     'weasely',
     'category',
     'laughable',
     'gramps',
     'safdar',
     'calorie',
     'scandi',
     'cannon',
     'maliciously',
     'bothered',
     'troi',
     'couleur',
     'visionary',
     'fizzles',
     'evangalizing',
     'reeves',
     'bombadier',
     'bowlegged',
     'custody',
     'weta',
     'archambault',
     'warlords',
     'makeout',
     'bonbons',
     'importances',
     'baruchel',
     'floyd',
     'infirm',
     'bloodwaters',
     'ashford',
     'colleagues',
     'discern',
     'thunderjet',
     'pullers',
     'evos',
     'celebrations',
     'seely',
     'nasty',
     'keach',
     'tonge',
     'senki',
     'approxiamtely',
     'unable',
     'rayburn',
     'britons',
     'christoph',
     'proctor',
     'tapped',
     'lenz',
     'vengeant',
     'exaggerating',
     'mle',
     'declaims',
     'hight',
     'repetoir',
     'yolu',
     'smarty',
     'steels',
     'openness',
     'coached',
     'archiving',
     'horrendous',
     'engages',
     'loosing',
     'anchorpoint',
     'fecal',
     'gracefully',
     'tapioca',
     'bizniss',
     'overhyped',
     'shortland',
     'cleansed',
     'negativity',
     'gushy',
     'mortitz',
     'stripper',
     'woke',
     'slayers',
     'uncensored',
     'textiles',
     'louda',
     'castrati',
     'altmanesque',
     'yes',
     'huntress',
     'urging',
     'tua',
     'sentient',
     'kellogg',
     'cheerful',
     'swanks',
     'shor',
     'cheapo',
     'flourishing',
     'tap',
     'kph',
     'bobbidi',
     'tangos',
     'honey',
     'oswald',
     'philippians',
     'payroll',
     'chooses',
     'archtypes',
     'generators',
     'grillo',
     'horrorible',
     'yellowing',
     'vancouver',
     'thet',
     'babtise',
     'participates',
     'uriah',
     'loust',
     'ravishingly',
     'punishing',
     'jhoom',
     'lulling',
     'stetting',
     'wierd',
     'truce',
     'peerce',
     'transpose',
     'unplanned',
     'unmistakeably',
     'approval',
     'amontillado',
     'een',
     'lefties',
     'tentatives',
     'mysteriousness',
     'mid',
     'technicians',
     'wich',
     'englund',
     'freespirited',
     'kun',
     'discourses',
     'nyily',
     'honorably',
     'hankerchief',
     'nugget',
     'nationalism',
     'reveals',
     'lamppost',
     'tempra',
     'sanctimoniousness',
     'wardrobes',
     'visa',
     'lenses',
     'johars',
     'prefers',
     'webster',
     'marcuzzo',
     'licensable',
     'brilliancy',
     'gumbas',
     'jacoby',
     'twine',
     'entices',
     'unpremeditated',
     'jin',
     'affirmatively',
     'joyful',
     'plotkurt',
     'danniele',
     'rpond',
     'flare',
     'lester',
     'toying',
     'having',
     'anorexia',
     'hoof',
     'stillman',
     'hows',
     'contrite',
     'hersholt',
     'utterance',
     'superflous',
     'orders',
     'pamelyn',
     'traumatized',
     'poder',
     'virtuality',
     'reaper',
     'trini',
     'phantasm',
     'fbp',
     'nuked',
     'siegfried',
     'ralph',
     'erwin',
     'rhymer',
     'christien',
     'sidekick',
     'grasshopper',
     'steryotypes',
     'donnagio',
     'denny',
     'fraudulent',
     'weisse',
     'yoji',
     'adapters',
     'andalthough',
     'fee',
     'attorney',
     'holliday',
     'prerequisite',
     'ives',
     'yvaine',
     'smaller',
     'satired',
     'ghillie',
     'hagelin',
     'upsurge',
     'empirical',
     'smap',
     'kirk',
     'conservatism',
     'wesley',
     'becuz',
     'fantasia',
     'treadstone',
     'berdalh',
     'reaganomics',
     'schwarzenberg',
     'housemann',
     'jumpstart',
     'glamorise',
     'braves',
     'simply',
     'which',
     'knifes',
     'ramblings',
     'bused',
     'lombardo',
     'refresher',
     'evenings',
     'openings',
     'rings',
     'reverend',
     'blurry',
     'baldy',
     'acing',
     'mollys',
     'meditteranean',
     'workday',
     'apologies',
     'empathise',
     'outs',
     'hmmmmmmmm',
     'enquiry',
     'detector',
     'copying',
     'outlive',
     'gangsta',
     'koyaanisqatsi',
     'entrenches',
     'author',
     'undistinguished',
     'izzard',
     'orgue',
     'negotiator',
     'behaviorally',
     'eyebrowed',
     'maximizes',
     'pilippinos',
     'recurred',
     'bullt',
     'infinnerty',
     'suspicious',
     'uncooked',
     'these',
     'ozaki',
     'sweden',
     'petition',
     'opium',
     'complacency',
     'deux',
     'kramer',
     'opt',
     'auras',
     'shyamalan',
     'lamore',
     'sunbathing',
     'toxins',
     'limned',
     'khali',
     'jefferey',
     'interviewee',
     'righted',
     'grandmammy',
     'wol',
     'verica',
     'footwork',
     'doug',
     'euthanasiarist',
     'repeating',
     'debutante',
     'trusts',
     'righto',
     'phyllida',
     'upa',
     'doogie',
     'gig',
     'violins',
     'ardor',
     'ould',
     'stymieing',
     'libs',
     'alejo',
     'sick',
     'propensities',
     'occasions',
     'spiderman',
     'limousines',
     'hearkening',
     'reinstated',
     'concede',
     'vineyard',
     'image',
     'waxed',
     'inuyasha',
     'paralyzed',
     'notches',
     'latifah',
     'mediation',
     'cozies',
     'spirit',
     'fathoms',
     'uecker',
     'hoochie',
     'akria',
     'praises',
     'wiring',
     'pastparticularly',
     'ghastliness',
     'artiness',
     'gruner',
     'admirals',
     'egger',
     'extract',
     'guiltlessly',
     'pie',
     'audaciousness',
     'stallonethat',
     'balconys',
     'cassi',
     'definable',
     'rote',
     'assaulted',
     'schmoeller',
     'cancer',
     'equality',
     'kruk',
     'whoah',
     'dalai',
     'tuareg',
     'split',
     'bollywood',
     'mates',
     'supports',
     'whiskers',
     'meres',
     'plasticine',
     'bartel',
     'phrase',
     'poldark',
     'pylon',
     'undefined',
     'videographer',
     'blithesome',
     'prendergast',
     'goddard',
     'spectular',
     'fof',
     'kiddie',
     'accelerating',
     'secreted',
     'manslaughter',
     'akimbo',
     'privacy',
     'michigan',
     'ambiguities',
     'belabors',
     'mol',
     'disemboweled',
     'creely',
     'nosebleed',
     'autobiography',
     'dispelled',
     'lancie',
     'revolutionaries',
     'allende',
     'jacy',
     'kostic',
     'tormei',
     'chiefly',
     'atmospheric',
     'europa',
     'judmila',
     'extremal',
     'decaprio',
     'amore',
     'cockneys',
     'chong',
     'coordinates',
     'ctomvelu',
     'scums',
     'valleyspeak',
     'minstrel',
     'shoddier',
     'combusted',
     'tirade',
     'marketplaces',
     'reflex',
     'rjt',
     'deckard',
     'godfathers',
     'sibling',
     'erupted',
     'wasnt',
     'lollipop',
     'narcotics',
     'showdowns',
     'excess',
     'taught',
     'persuade',
     'homer',
     'binysh',
     'ravaging',
     'minutest',
     'yomada',
     'leckie',
     'snazzy',
     'rafting',
     'grendelif',
     'nemeses',
     'westmore',
     'sty',
     'puertorricans',
     'zaara',
     'timemachine',
     'similarities',
     'colera',
     'firefall',
     'winked',
     'painkiller',
     'leaflets',
     'tehran',
     'hooker',
     'appalingly',
     'humility',
     'illegitimate',
     'coer',
     'responisible',
     'conceded',
     'scarves',
     'dawid',
     'overflows',
     'annuder',
     'nickelodean',
     'comanche',
     'betrail',
     'pillage',
     'daffy',
     'dobson',
     'tessier',
     'egoism',
     'meanie',
     'trancers',
     'sequences',
     'viciente',
     'redlich',
     'filmfrderung',
     'leveled',
     'performer',
     'opponent',
     'appears',
     'squeaks',
     'peripheral',
     'blimey',
     'glass',
     'captors',
     'strains',
     'codenamealexa',
     'tooo',
     'aiello',
     'matines',
     'calibre',
     'tighten',
     'papercuts',
     'necrotic',
     'hums',
     'kavner',
     'employers',
     'troy',
     'almerayeda',
     'barnet',
     'nicotero',
     'rush',
     'ahehehe',
     'dui',
     'bleeps',
     'heroe',
     'gangreen',
     'paintbrush',
     'dowager',
     'khakkee',
     'chariots',
     'benfer',
     'mcneely',
     'quelled',
     'blockheads',
     'dufy',
     'badmen',
     'dondaro',
     'nachoo',
     'intercedes',
     'looksand',
     'hasidic',
     'will',
     'practicable',
     'reading',
     'manufacture',
     'bao',
     'cigarette',
     'chomps',
     'subverting',
     'reichdeutch',
     'dexter',
     'hrishitta',
     'splitting',
     'uproarious',
     'ametuer',
     'speedway',
     'worser',
     'brisco',
     'stream',
     'etre',
     'lengths',
     'chimpnaut',
     'corny',
     'stirring',
     'tremendous',
     'tually',
     'mnage',
     'ashitaka',
     'crossbows',
     'hackery',
     'riker',
     'twelve',
     'freshner',
     'bobbie',
     'percussion',
     'overpopulation',
     'eeeekkk',
     'centaury',
     'summitting',
     'andbest',
     'pumping',
     'somnolent',
     'infatuation',
     'shakesphere',
     'ingred',
     'moon',
     'keven',
     'sanguisuga',
     'quivers',
     'equalling',
     'vaugely',
     'supervising',
     'dissolved',
     'cheshire',
     'retribution',
     'cartoons',
     'maisie',
     'reptiles',
     'rsther',
     'erratically',
     'hoyt',
     ...]




```python
import numpy as np

layer_0 = np.zeros((1,vocab_size))
layer_0
```




    array([[ 0.,  0.,  0., ...,  0.,  0.,  0.]])




```python
from IPython.display import Image
Image(filename='sentiment_network.png')
```




![png](/assets/img/neural_network/output_24_0.png)




```python
word2index = {}

for i,word in enumerate(vocab):
    word2index[word] = i
word2index
```




    {'': 0,
     'werewoves': 1,
     'endowments': 2,
     'palace': 3,
     'persiflage': 4,
     'slasherville': 5,
     'locally': 6,
     'unrecycled': 7,
     'spearhead': 8,
     'allyson': 9,
     'manhating': 10,
     'bartok': 11,
     'gretorexes': 12,
     'soaks': 13,
     'protestations': 14,
     'superimposes': 15,
     'theirry': 16,
     'yaqui': 17,
     'contrives': 18,
     'accessorizing': 19,
     'arg': 20,
     'sanguine': 21,
     'batouch': 22,
     'asked': 23,
     'animals': 24,
     'cockpits': 25,
     'gorilla': 26,
     'diculous': 27,
     'establishing': 28,
     'kagemusha': 29,
     'sketches': 30,
     'rebuilt': 31,
     'perniciously': 32,
     'socioeconomic': 33,
     'ladylike': 34,
     'prognostication': 35,
     'blech': 36,
     'sugarbabe': 37,
     'desk': 38,
     'fez': 39,
     'accents': 40,
     'speach': 41,
     'rooster': 42,
     'effort': 43,
     'bodega': 44,
     'dong': 45,
     'preordained': 46,
     'dubliners': 47,
     'vili': 48,
     'imperatives': 49,
     'artifices': 50,
     'wieder': 51,
     'climate': 52,
     'whoopdedoodles': 53,
     'quatermass': 54,
     'inveterate': 55,
     'memorandum': 56,
     'crucially': 57,
     'bulimics': 58,
     'misdrawing': 59,
     'plympton': 60,
     'fireballs': 61,
     'verdant': 62,
     'testi': 63,
     'undeservingly': 64,
     'lusted': 65,
     'shylock': 66,
     'disinfecting': 67,
     'boxer': 68,
     'givney': 69,
     'hs': 70,
     'loser': 71,
     'civics': 72,
     'volcano': 73,
     'jur': 74,
     'mohnish': 75,
     'candidates': 76,
     'assemble': 77,
     'simi': 78,
     'resort': 79,
     'hessling': 80,
     'starbase': 81,
     'orgolini': 82,
     'starrett': 83,
     'weaker': 84,
     'transcending': 85,
     'levitate': 86,
     'spurns': 87,
     'contradictory': 88,
     'cambreau': 89,
     'latvia': 90,
     'kirkpatrick': 91,
     'betty': 92,
     'agnostic': 93,
     'sosa': 94,
     'kanji': 95,
     'swill': 96,
     'millinium': 97,
     'macgregor': 98,
     'brd': 99,
     'ariete': 100,
     'assassins': 101,
     'disscusion': 102,
     'legislative': 103,
     'dwars': 104,
     'controller': 105,
     'hadass': 106,
     'vega': 107,
     'bends': 108,
     'glock': 109,
     'spacewalk': 110,
     'va': 111,
     'offa': 112,
     'winfield': 113,
     'somewhat': 114,
     'yates': 115,
     'vinyl': 116,
     'complicity': 117,
     'bela': 118,
     'squishes': 119,
     'rippings': 120,
     'eyed': 121,
     'amatuerish': 122,
     'desilva': 123,
     'christmass': 124,
     'briley': 125,
     'bakhtyari': 126,
     'unmasked': 127,
     'huffman': 128,
     'fallacious': 129,
     'problem': 130,
     'sieger': 131,
     'koma': 132,
     'grovelling': 133,
     'incl': 134,
     'farlinger': 135,
     'teasers': 136,
     'huff': 137,
     'untried': 138,
     'crocker': 139,
     'dansu': 140,
     'scammers': 141,
     'popsicle': 142,
     'arthritic': 143,
     'grubs': 144,
     'exemplar': 145,
     'racial': 146,
     'verbiage': 147,
     'saloshin': 148,
     'painlessly': 149,
     'harewood': 150,
     'shart': 151,
     'keepers': 152,
     'archrivals': 153,
     'longish': 154,
     'batmobile': 155,
     'shakespearian': 156,
     'bestselling': 157,
     'spewing': 158,
     'midlands': 159,
     'trattoria': 160,
     'greenaway': 161,
     'gestapo': 162,
     'ed': 163,
     'huns': 164,
     'bloch': 165,
     'mashall': 166,
     'versy': 167,
     'david': 168,
     'sicilian': 169,
     'propositioned': 170,
     'eighty': 171,
     'carridine': 172,
     'delicates': 173,
     'veering': 174,
     'columbus': 175,
     'dunning': 176,
     'mercantile': 177,
     'rape': 178,
     'purely': 179,
     'rediscovered': 180,
     'abstinence': 181,
     'clunes': 182,
     'emerson': 183,
     'judgments': 184,
     'lawful': 185,
     'celebration': 186,
     'affirmative': 187,
     'sedately': 188,
     'sng': 189,
     'inuindo': 190,
     'mosely': 191,
     'bungalow': 192,
     'ninga': 193,
     'dripped': 194,
     'itallian': 195,
     'himalaya': 196,
     'shikoku': 197,
     'braik': 198,
     'grousing': 199,
     'nair': 200,
     'forrester': 201,
     'elemental': 202,
     'allegations': 203,
     'delilah': 204,
     'boneheaded': 205,
     'baltimoreans': 206,
     'dunebuggies': 207,
     'taguchi': 208,
     'coleseum': 209,
     'saratoga': 210,
     'ninotchka': 211,
     'afganistan': 212,
     'genorisity': 213,
     'haff': 214,
     'jennilee': 215,
     'jesues': 216,
     'dwarfs': 217,
     'enchilada': 218,
     'feminist': 219,
     'ghettoisation': 220,
     'handlebar': 221,
     'antagonistic': 222,
     'marian': 223,
     'crichton': 224,
     'ryo': 225,
     'mean': 226,
     'inheritance': 227,
     'presently': 228,
     'pear': 229,
     'inequality': 230,
     'stately': 231,
     'nooo': 232,
     'obscurities': 233,
     'determinedly': 234,
     'solemn': 235,
     'sullenly': 236,
     'machism': 237,
     'tingled': 238,
     'maschera': 239,
     'tristran': 240,
     'mendoza': 241,
     'baked': 242,
     'jonatha': 243,
     'lowly': 244,
     'halliwell': 245,
     'msted': 246,
     'rodann': 247,
     'hunkered': 248,
     'cashmere': 249,
     'chevalia': 250,
     'jakub': 251,
     'dobermann': 252,
     'overexxagerating': 253,
     'pfennig': 254,
     'fisted': 255,
     'mcelwee': 256,
     'chief': 257,
     'parlor': 258,
     'browbeating': 259,
     'parasol': 260,
     'negligible': 261,
     'kira': 262,
     'monceau': 263,
     'blew': 264,
     'odete': 265,
     'muco': 266,
     'predominantly': 267,
     'levon': 268,
     'discourage': 269,
     'fragmented': 270,
     'vandermey': 271,
     'etude': 272,
     'mitch': 273,
     'sandbag': 274,
     'bending': 275,
     'dizzying': 276,
     'mover': 277,
     'rewired': 278,
     'awww': 279,
     'di': 280,
     'bejesus': 281,
     'wallet': 282,
     'uprooting': 283,
     'atari': 284,
     'dreamlike': 285,
     'exacted': 286,
     'harbouring': 287,
     'indiscreet': 288,
     'turks': 289,
     'gems': 290,
     'hoboken': 291,
     'yalom': 292,
     'rooftop': 293,
     'howit': 294,
     'tolson': 295,
     'tulane': 296,
     'reductive': 297,
     'catharthic': 298,
     'famarialy': 299,
     'sista': 300,
     'ghidorah': 301,
     'ngoombujarra': 302,
     'intently': 303,
     'jlu': 304,
     'kyrptonite': 305,
     'hilda': 306,
     'census': 307,
     'baguette': 308,
     'mondrians': 309,
     'advisable': 310,
     'mcnee': 311,
     'candlelit': 312,
     'affability': 313,
     'intercut': 314,
     'installations': 315,
     'elliptical': 316,
     'washer': 317,
     'colt': 318,
     'pevensie': 319,
     'outshined': 320,
     'despotic': 321,
     'suares': 322,
     'privates': 323,
     'scrabble': 324,
     'milliardo': 325,
     'booting': 326,
     'rowan': 327,
     'golmaal': 328,
     'pueblos': 329,
     'msf': 330,
     'giulietta': 331,
     'phili': 332,
     'pleaseee': 333,
     'connecticute': 334,
     'rosnelski': 335,
     'tenebra': 336,
     'bako': 337,
     'blessings': 338,
     'smudge': 339,
     'cya': 340,
     'pummel': 341,
     'brocks': 342,
     'homere': 343,
     'propellant': 344,
     'deliveried': 345,
     'finisham': 346,
     'newsradio': 347,
     'bernie': 348,
     'gouden': 349,
     'enchant': 350,
     'bessie': 351,
     'semisubmerged': 352,
     'extraterrestrial': 353,
     'believably': 354,
     'accomplice': 355,
     'dooku': 356,
     'baja': 357,
     'met': 358,
     'circulate': 359,
     'disobeyed': 360,
     'quakerly': 361,
     'overstyling': 362,
     'softens': 363,
     'units': 364,
     'shaye': 365,
     'starters': 366,
     'gripes': 367,
     'nightmarish': 368,
     'patriotic': 369,
     'goodtimes': 370,
     'stroheim': 371,
     'debit': 372,
     'prissies': 373,
     'woebegone': 374,
     'deputies': 375,
     'awkwardness': 376,
     'obama': 377,
     'tarazu': 378,
     'kendra': 379,
     'patriots': 380,
     'helpfuls': 381,
     'mightily': 382,
     'polemical': 383,
     'unruly': 384,
     'planing': 385,
     'paperhouse': 386,
     'sororities': 387,
     'pym': 388,
     'therin': 389,
     'tarkovsky': 390,
     'rdiger': 391,
     'resembling': 392,
     'gimmicks': 393,
     'iler': 394,
     'lineal': 395,
     'taming': 396,
     'mortenson': 397,
     'waugh': 398,
     'furies': 399,
     'grufford': 400,
     'hammill': 401,
     'plunkett': 402,
     'paterson': 403,
     'konishita': 404,
     'immorality': 405,
     'angelos': 406,
     'kebbel': 407,
     'tamiroff': 408,
     'boen': 409,
     'rivalry': 410,
     'ethnic': 411,
     'funner': 412,
     'troops': 413,
     'deadeningly': 414,
     'watcha': 415,
     'dhiraj': 416,
     'haranguing': 417,
     'dejas': 418,
     'weasely': 419,
     'category': 420,
     'laughable': 421,
     'gramps': 422,
     'safdar': 423,
     'calorie': 424,
     'scandi': 425,
     'cannon': 426,
     'maliciously': 427,
     'bothered': 428,
     'troi': 429,
     'couleur': 430,
     'visionary': 431,
     'fizzles': 432,
     'evangalizing': 433,
     'reeves': 434,
     'bombadier': 435,
     'bowlegged': 436,
     'custody': 437,
     'weta': 438,
     'archambault': 439,
     'warlords': 440,
     'makeout': 441,
     'bonbons': 442,
     'importances': 443,
     'baruchel': 444,
     'floyd': 445,
     'infirm': 446,
     'bloodwaters': 447,
     'ashford': 448,
     'colleagues': 449,
     'discern': 450,
     'thunderjet': 451,
     'pullers': 452,
     'evos': 453,
     'celebrations': 454,
     'seely': 455,
     'nasty': 456,
     'keach': 457,
     'tonge': 458,
     'senki': 459,
     'approxiamtely': 460,
     'unable': 461,
     'rayburn': 462,
     'britons': 463,
     'christoph': 464,
     'proctor': 465,
     'tapped': 466,
     'lenz': 467,
     'vengeant': 468,
     'exaggerating': 469,
     'mle': 470,
     'declaims': 471,
     'hight': 472,
     'repetoir': 473,
     'yolu': 474,
     'smarty': 475,
     'steels': 476,
     'openness': 477,
     'coached': 478,
     'archiving': 479,
     'horrendous': 480,
     'engages': 481,
     'loosing': 482,
     'anchorpoint': 483,
     'fecal': 484,
     'gracefully': 485,
     'tapioca': 486,
     'bizniss': 487,
     'overhyped': 488,
     'shortland': 489,
     'cleansed': 490,
     'negativity': 491,
     'gushy': 492,
     'mortitz': 493,
     'stripper': 494,
     'woke': 495,
     'slayers': 496,
     'uncensored': 497,
     'textiles': 498,
     'louda': 499,
     'castrati': 500,
     'altmanesque': 501,
     'yes': 502,
     'huntress': 503,
     'urging': 504,
     'tua': 505,
     'sentient': 506,
     'kellogg': 507,
     'cheerful': 508,
     'swanks': 509,
     'shor': 510,
     'cheapo': 511,
     'flourishing': 512,
     'tap': 513,
     'kph': 514,
     'bobbidi': 515,
     'tangos': 516,
     'honey': 517,
     'oswald': 518,
     'philippians': 519,
     'payroll': 520,
     'chooses': 521,
     'archtypes': 522,
     'generators': 523,
     'grillo': 524,
     'horrorible': 525,
     'yellowing': 526,
     'vancouver': 527,
     'thet': 528,
     'babtise': 529,
     'participates': 530,
     'uriah': 531,
     'loust': 532,
     'ravishingly': 533,
     'punishing': 534,
     'jhoom': 535,
     'lulling': 536,
     'stetting': 537,
     'wierd': 538,
     'truce': 539,
     'peerce': 540,
     'transpose': 541,
     'unplanned': 542,
     'unmistakeably': 543,
     'approval': 544,
     'amontillado': 545,
     'een': 546,
     'lefties': 547,
     'tentatives': 548,
     'mysteriousness': 549,
     'mid': 550,
     'technicians': 551,
     'wich': 552,
     'englund': 553,
     'freespirited': 554,
     'kun': 555,
     'discourses': 556,
     'nyily': 557,
     'honorably': 558,
     'hankerchief': 559,
     'nugget': 560,
     'nationalism': 561,
     'reveals': 562,
     'lamppost': 563,
     'tempra': 564,
     'sanctimoniousness': 565,
     'wardrobes': 566,
     'visa': 567,
     'lenses': 568,
     'johars': 569,
     'prefers': 570,
     'webster': 571,
     'marcuzzo': 572,
     'licensable': 573,
     'brilliancy': 574,
     'gumbas': 575,
     'jacoby': 576,
     'twine': 577,
     'entices': 578,
     'unpremeditated': 579,
     'jin': 580,
     'affirmatively': 581,
     'joyful': 582,
     'plotkurt': 583,
     'danniele': 584,
     'rpond': 585,
     'flare': 586,
     'lester': 587,
     'toying': 588,
     'having': 589,
     'anorexia': 590,
     'hoof': 591,
     'stillman': 592,
     'hows': 593,
     'contrite': 594,
     'hersholt': 595,
     'utterance': 596,
     'superflous': 597,
     'orders': 598,
     'pamelyn': 599,
     'traumatized': 600,
     'poder': 601,
     'virtuality': 602,
     'reaper': 603,
     'trini': 604,
     'phantasm': 605,
     'fbp': 606,
     'nuked': 607,
     'siegfried': 608,
     'ralph': 609,
     'erwin': 610,
     'rhymer': 611,
     'christien': 612,
     'sidekick': 613,
     'grasshopper': 614,
     'steryotypes': 615,
     'donnagio': 616,
     'denny': 617,
     'fraudulent': 618,
     'weisse': 619,
     'yoji': 620,
     'adapters': 621,
     'andalthough': 622,
     'fee': 623,
     'attorney': 624,
     'holliday': 625,
     'prerequisite': 626,
     'ives': 627,
     'yvaine': 628,
     'smaller': 629,
     'satired': 630,
     'ghillie': 631,
     'hagelin': 632,
     'upsurge': 633,
     'empirical': 634,
     'smap': 635,
     'kirk': 636,
     'conservatism': 637,
     'wesley': 638,
     'becuz': 639,
     'fantasia': 640,
     'treadstone': 641,
     'berdalh': 642,
     'reaganomics': 643,
     'schwarzenberg': 644,
     'housemann': 645,
     'jumpstart': 646,
     'glamorise': 647,
     'braves': 648,
     'simply': 649,
     'which': 650,
     'knifes': 651,
     'ramblings': 652,
     'bused': 653,
     'lombardo': 654,
     'refresher': 655,
     'evenings': 656,
     'openings': 657,
     'rings': 658,
     'reverend': 659,
     'blurry': 660,
     'baldy': 661,
     'acing': 662,
     'mollys': 663,
     'meditteranean': 664,
     'workday': 665,
     'apologies': 666,
     'empathise': 667,
     'outs': 668,
     'hmmmmmmmm': 669,
     'enquiry': 670,
     'detector': 671,
     'copying': 672,
     'outlive': 673,
     'gangsta': 674,
     'koyaanisqatsi': 675,
     'entrenches': 676,
     'author': 677,
     'undistinguished': 678,
     'izzard': 679,
     'orgue': 680,
     'negotiator': 681,
     'behaviorally': 682,
     'eyebrowed': 683,
     'maximizes': 684,
     'pilippinos': 685,
     'recurred': 686,
     'bullt': 687,
     'infinnerty': 688,
     'suspicious': 689,
     'uncooked': 690,
     'these': 691,
     'ozaki': 692,
     'sweden': 693,
     'petition': 694,
     'opium': 695,
     'complacency': 696,
     'deux': 697,
     'kramer': 698,
     'opt': 699,
     'auras': 700,
     'shyamalan': 701,
     'lamore': 702,
     'sunbathing': 703,
     'toxins': 704,
     'limned': 705,
     'khali': 706,
     'jefferey': 707,
     'interviewee': 708,
     'righted': 709,
     'grandmammy': 710,
     'wol': 711,
     'verica': 712,
     'footwork': 713,
     'doug': 714,
     'euthanasiarist': 715,
     'repeating': 716,
     'debutante': 717,
     'trusts': 718,
     'righto': 719,
     'phyllida': 720,
     'upa': 721,
     'doogie': 722,
     'gig': 723,
     'violins': 724,
     'ardor': 725,
     'ould': 726,
     'stymieing': 727,
     'libs': 728,
     'alejo': 729,
     'sick': 730,
     'propensities': 731,
     'occasions': 732,
     'spiderman': 733,
     'limousines': 734,
     'hearkening': 735,
     'reinstated': 736,
     'concede': 737,
     'vineyard': 738,
     'image': 739,
     'waxed': 740,
     'inuyasha': 741,
     'paralyzed': 742,
     'notches': 743,
     'latifah': 744,
     'mediation': 745,
     'cozies': 746,
     'spirit': 747,
     'fathoms': 748,
     'uecker': 749,
     'hoochie': 750,
     'akria': 751,
     'praises': 752,
     'wiring': 753,
     'pastparticularly': 754,
     'ghastliness': 755,
     'artiness': 756,
     'gruner': 757,
     'admirals': 758,
     'egger': 759,
     'extract': 760,
     'guiltlessly': 761,
     'pie': 762,
     'audaciousness': 763,
     'stallonethat': 764,
     'balconys': 765,
     'cassi': 766,
     'definable': 767,
     'rote': 768,
     'assaulted': 769,
     'schmoeller': 770,
     'cancer': 771,
     'equality': 772,
     'kruk': 773,
     'whoah': 774,
     'dalai': 775,
     'tuareg': 776,
     'split': 777,
     'bollywood': 778,
     'mates': 779,
     'supports': 780,
     'whiskers': 781,
     'meres': 782,
     'plasticine': 783,
     'bartel': 784,
     'phrase': 785,
     'poldark': 786,
     'pylon': 787,
     'undefined': 788,
     'videographer': 789,
     'blithesome': 790,
     'prendergast': 791,
     'goddard': 792,
     'spectular': 793,
     'fof': 794,
     'kiddie': 795,
     'accelerating': 796,
     'secreted': 797,
     'manslaughter': 798,
     'akimbo': 799,
     'privacy': 800,
     'michigan': 801,
     'ambiguities': 802,
     'belabors': 803,
     'mol': 804,
     'disemboweled': 805,
     'creely': 806,
     'nosebleed': 807,
     'autobiography': 808,
     'dispelled': 809,
     'lancie': 810,
     'revolutionaries': 811,
     'allende': 812,
     'jacy': 813,
     'kostic': 814,
     'tormei': 815,
     'chiefly': 816,
     'atmospheric': 817,
     'europa': 818,
     'judmila': 819,
     'extremal': 820,
     'decaprio': 821,
     'amore': 822,
     'cockneys': 823,
     'chong': 824,
     'coordinates': 825,
     'ctomvelu': 826,
     'scums': 827,
     'valleyspeak': 828,
     'minstrel': 829,
     'shoddier': 830,
     'combusted': 831,
     'tirade': 832,
     'marketplaces': 833,
     'reflex': 834,
     'rjt': 835,
     'deckard': 836,
     'godfathers': 837,
     'sibling': 838,
     'erupted': 839,
     'wasnt': 840,
     'lollipop': 841,
     'narcotics': 842,
     'showdowns': 843,
     'excess': 844,
     'taught': 845,
     'persuade': 846,
     'homer': 847,
     'binysh': 848,
     'ravaging': 849,
     'minutest': 850,
     'yomada': 851,
     'leckie': 852,
     'snazzy': 853,
     'rafting': 854,
     'grendelif': 855,
     'nemeses': 856,
     'westmore': 857,
     'sty': 858,
     'puertorricans': 859,
     'zaara': 860,
     'timemachine': 861,
     'similarities': 862,
     'colera': 863,
     'firefall': 864,
     'winked': 865,
     'painkiller': 866,
     'leaflets': 867,
     'tehran': 868,
     'hooker': 869,
     'appalingly': 870,
     'humility': 871,
     'illegitimate': 872,
     'coer': 873,
     'responisible': 874,
     'conceded': 875,
     'scarves': 876,
     'dawid': 877,
     'overflows': 878,
     'annuder': 879,
     'nickelodean': 880,
     'comanche': 881,
     'betrail': 882,
     'pillage': 883,
     'daffy': 884,
     'dobson': 885,
     'tessier': 886,
     'egoism': 887,
     'meanie': 888,
     'trancers': 889,
     'sequences': 890,
     'viciente': 891,
     'redlich': 892,
     'filmfrderung': 893,
     'leveled': 894,
     'performer': 895,
     'opponent': 896,
     'appears': 897,
     'squeaks': 898,
     'peripheral': 899,
     'blimey': 900,
     'glass': 901,
     'captors': 902,
     'strains': 903,
     'codenamealexa': 904,
     'tooo': 905,
     'aiello': 906,
     'matines': 907,
     'calibre': 908,
     'tighten': 909,
     'papercuts': 910,
     'necrotic': 911,
     'hums': 912,
     'kavner': 913,
     'employers': 914,
     'troy': 915,
     'almerayeda': 916,
     'barnet': 917,
     'nicotero': 918,
     'rush': 919,
     'ahehehe': 920,
     'dui': 921,
     'bleeps': 922,
     'heroe': 923,
     'gangreen': 924,
     'paintbrush': 925,
     'dowager': 926,
     'khakkee': 927,
     'chariots': 928,
     'benfer': 929,
     'mcneely': 930,
     'quelled': 931,
     'blockheads': 932,
     'dufy': 933,
     'badmen': 934,
     'dondaro': 935,
     'nachoo': 936,
     'intercedes': 937,
     'looksand': 938,
     'hasidic': 939,
     'will': 940,
     'practicable': 941,
     'reading': 942,
     'manufacture': 943,
     'bao': 944,
     'cigarette': 945,
     'chomps': 946,
     'subverting': 947,
     'reichdeutch': 948,
     'dexter': 949,
     'hrishitta': 950,
     'splitting': 951,
     'uproarious': 952,
     'ametuer': 953,
     'speedway': 954,
     'worser': 955,
     'brisco': 956,
     'stream': 957,
     'etre': 958,
     'lengths': 959,
     'chimpnaut': 960,
     'corny': 961,
     'stirring': 962,
     'tremendous': 963,
     'tually': 964,
     'mnage': 965,
     'ashitaka': 966,
     'crossbows': 967,
     'hackery': 968,
     'riker': 969,
     'twelve': 970,
     'freshner': 971,
     'bobbie': 972,
     'percussion': 973,
     'overpopulation': 974,
     'eeeekkk': 975,
     'centaury': 976,
     'summitting': 977,
     'andbest': 978,
     'pumping': 979,
     'somnolent': 980,
     'infatuation': 981,
     'shakesphere': 982,
     'ingred': 983,
     'moon': 984,
     'keven': 985,
     'sanguisuga': 986,
     'quivers': 987,
     'equalling': 988,
     'vaugely': 989,
     'supervising': 990,
     'dissolved': 991,
     'cheshire': 992,
     'retribution': 993,
     'cartoons': 994,
     'maisie': 995,
     'reptiles': 996,
     'rsther': 997,
     'erratically': 998,
     'hoyt': 999,
     ...}




```python
def update_input_layer(review):
    
    global layer_0
    
    # clear out previous state, reset the layer to be all 0s
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

update_input_layer(reviews[0])
```


```python
layer_0
```




    array([[ 18.,   0.,   0., ...,   0.,   0.,   0.]])




```python
def get_target_for_label(label):
    if(label == 'POSITIVE'):
        return 1
    else:
        return 0
```


```python
labels[0]
```




    'POSITIVE'




```python
get_target_for_label(labels[0])
```




    1




```python
labels[1]
```




    'NEGATIVE'




```python
get_target_for_label(labels[1])
```




    0



# Project 3: Building a Neural Network

- Start with your neural network from the last chapter
- 3 layer neural network
- no non-linearity in hidden layer
- use our functions to create the training data
- create a "pre_process_data" function to create vocabulary for our training data generating functions
- modify "train" to train over the entire corpus

### Where to Get Help if You Need it
- Re-watch previous week's Udacity Lectures
- Chapters 3-5 - [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) - (40% Off: **traskud17**)


```python
import time
import sys
import numpy as np

# Let's tweak our network from before to model these phenomena
class SentimentNetwork:
    def __init__(self, reviews,labels,hidden_nodes = 10, learning_rate = 0.1):
       
        # set our random number generator 
        np.random.seed(1)
    
        self.pre_process_data(reviews, labels)
        
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)
        
        
    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1,input_nodes))
    
        
    def update_input_layer(self,review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(" "):
            if(word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] += 1
                
    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def train(self, training_reviews, training_labels):
        
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer
            self.update_input_layer(review)

            # Hidden layer
            layer_1 = self.layer_0.dot(self.weights_0_1)

            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # TODO: Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # TODO: Update the weights
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate # update input-to-hidden weights with gradient descent step

            if(np.abs(layer_2_error) < 0.5):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                            + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        
        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
        
```


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
```


```python
# evaluate our model before training (just to show how horrible it is)
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):1242.% #Correct:500 #Tested:1000 Testing Accuracy:50.0%


```python
# train the network
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:0.0% Speed(reviews/sec):0.0 #Correct:0 #Trained:1 Training Accuracy:0.0%
    Progress:10.4% Speed(reviews/sec):167.7 #Correct:1250 #Trained:2501 Training Accuracy:49.9%
    Progress:20.8% Speed(reviews/sec):170.2 #Correct:2500 #Trained:5001 Training Accuracy:49.9%
    Progress:31.2% Speed(reviews/sec):169.9 #Correct:3750 #Trained:7501 Training Accuracy:49.9%
    Progress:41.6% Speed(reviews/sec):171.3 #Correct:5000 #Trained:10001 Training Accuracy:49.9%
    Progress:52.0% Speed(reviews/sec):170.0 #Correct:6250 #Trained:12501 Training Accuracy:49.9%
    Progress:62.5% Speed(reviews/sec):170.8 #Correct:7500 #Trained:15001 Training Accuracy:49.9%
    Progress:72.9% Speed(reviews/sec):171.4 #Correct:8750 #Trained:17501 Training Accuracy:49.9%
    Progress:83.3% Speed(reviews/sec):171.7 #Correct:10000 #Trained:20001 Training Accuracy:49.9%
    Progress:93.7% Speed(reviews/sec):172.6 #Correct:11250 #Trained:22501 Training Accuracy:49.9%
    Progress:99.9% Speed(reviews/sec):172.5 #Correct:11999 #Trained:24000 Training Accuracy:49.9%


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.01)
```


```python
# train the network
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:0.0% Speed(reviews/sec):0.0 #Correct:0 #Trained:1 Training Accuracy:0.0%
    Progress:10.4% Speed(reviews/sec):149.0 #Correct:1247 #Trained:2501 Training Accuracy:49.8%
    Progress:20.8% Speed(reviews/sec):145.3 #Correct:2497 #Trained:5001 Training Accuracy:49.9%
    Progress:31.2% Speed(reviews/sec):144.0 #Correct:3747 #Trained:7501 Training Accuracy:49.9%
    Progress:41.6% Speed(reviews/sec):141.8 #Correct:4997 #Trained:10001 Training Accuracy:49.9%
    Progress:52.0% Speed(reviews/sec):137.0 #Correct:6247 #Trained:12501 Training Accuracy:49.9%
    Progress:62.5% Speed(reviews/sec):137.7 #Correct:7489 #Trained:15001 Training Accuracy:49.9%
    Progress:72.9% Speed(reviews/sec):137.1 #Correct:8740 #Trained:17501 Training Accuracy:49.9%
    Progress:83.3% Speed(reviews/sec):138.1 #Correct:9990 #Trained:20001 Training Accuracy:49.9%
    Progress:93.7% Speed(reviews/sec):138.9 #Correct:11240 #Trained:22501 Training Accuracy:49.9%
    Progress:99.9% Speed(reviews/sec):139.4 #Correct:11989 #Trained:24000 Training Accuracy:49.9%


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)
```


```python
# train the network
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:0.0% Speed(reviews/sec):0.0 #Correct:0 #Trained:1 Training Accuracy:0.0%
    Progress:10.4% Speed(reviews/sec):147.5 #Correct:1267 #Trained:2501 Training Accuracy:50.6%
    Progress:20.8% Speed(reviews/sec):147.3 #Correct:2608 #Trained:5001 Training Accuracy:52.1%
    Progress:31.2% Speed(reviews/sec):147.3 #Correct:4021 #Trained:7501 Training Accuracy:53.6%
    Progress:41.6% Speed(reviews/sec):147.3 #Correct:5497 #Trained:10001 Training Accuracy:54.9%
    Progress:52.0% Speed(reviews/sec):147.3 #Correct:7071 #Trained:12501 Training Accuracy:56.5%
    Progress:62.5% Speed(reviews/sec):146.9 #Correct:8632 #Trained:15001 Training Accuracy:57.5%
    Progress:72.9% Speed(reviews/sec):146.9 #Correct:10228 #Trained:17501 Training Accuracy:58.4%
    Progress:83.3% Speed(reviews/sec):146.9 #Correct:11880 #Trained:20001 Training Accuracy:59.3%
    Progress:93.7% Speed(reviews/sec):147.0 #Correct:13580 #Trained:22501 Training Accuracy:60.3%
    Progress:99.9% Speed(reviews/sec):146.9 #Correct:14658 #Trained:24000 Training Accuracy:61.0%

# Understanding Neural Noise


```python
from IPython.display import Image
Image(filename='sentiment_network.png')
```




![png](/assets/img/neural_network/output_45_0.png)




```python
def update_input_layer(review):
    
    global layer_0
    
    # clear out previous state, reset the layer to be all 0s
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

update_input_layer(reviews[0])
```


```python
layer_0
```




    array([[ 18.,   0.,   0., ...,   0.,   0.,   0.]])




```python
review_counter = Counter()
```


```python
for word in reviews[0].split(" "):
    review_counter[word] += 1
```


```python
review_counter.most_common()
```




    [('.', 27),
     ('', 18),
     ('the', 9),
     ('to', 6),
     ('high', 5),
     ('i', 5),
     ('bromwell', 4),
     ('is', 4),
     ('a', 4),
     ('teachers', 4),
     ('that', 4),
     ('of', 4),
     ('it', 2),
     ('at', 2),
     ('as', 2),
     ('school', 2),
     ('my', 2),
     ('in', 2),
     ('me', 2),
     ('students', 2),
     ('their', 2),
     ('student', 2),
     ('cartoon', 1),
     ('comedy', 1),
     ('ran', 1),
     ('same', 1),
     ('time', 1),
     ('some', 1),
     ('other', 1),
     ('programs', 1),
     ('about', 1),
     ('life', 1),
     ('such', 1),
     ('years', 1),
     ('teaching', 1),
     ('profession', 1),
     ('lead', 1),
     ('believe', 1),
     ('s', 1),
     ('satire', 1),
     ('much', 1),
     ('closer', 1),
     ('reality', 1),
     ('than', 1),
     ('scramble', 1),
     ('survive', 1),
     ('financially', 1),
     ('insightful', 1),
     ('who', 1),
     ('can', 1),
     ('see', 1),
     ('right', 1),
     ('through', 1),
     ('pathetic', 1),
     ('pomp', 1),
     ('pettiness', 1),
     ('whole', 1),
     ('situation', 1),
     ('all', 1),
     ('remind', 1),
     ('schools', 1),
     ('knew', 1),
     ('and', 1),
     ('when', 1),
     ('saw', 1),
     ('episode', 1),
     ('which', 1),
     ('repeatedly', 1),
     ('tried', 1),
     ('burn', 1),
     ('down', 1),
     ('immediately', 1),
     ('recalled', 1),
     ('classic', 1),
     ('line', 1),
     ('inspector', 1),
     ('m', 1),
     ('here', 1),
     ('sack', 1),
     ('one', 1),
     ('your', 1),
     ('welcome', 1),
     ('expect', 1),
     ('many', 1),
     ('adults', 1),
     ('age', 1),
     ('think', 1),
     ('far', 1),
     ('fetched', 1),
     ('what', 1),
     ('pity', 1),
     ('isn', 1),
     ('t', 1)]



# Project 4: Reducing Noise in our Input Data


```python
import time
import sys
import numpy as np

# Let's tweak our network from before to model these phenomena
class SentimentNetwork:
    def __init__(self, reviews,labels,hidden_nodes = 10, learning_rate = 0.1):
       
        # set our random number generator 
        np.random.seed(1)
    
        self.pre_process_data(reviews, labels)
        
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)
        
        
    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1,input_nodes))
    
        
    def update_input_layer(self,review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(" "):
            if(word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] = 1
                
    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def train(self, training_reviews, training_labels):
        
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer
            self.update_input_layer(review)

            # Hidden layer
            layer_1 = self.layer_0.dot(self.weights_0_1)

            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # TODO: Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # TODO: Update the weights
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate # update input-to-hidden weights with gradient descent step

            if(np.abs(layer_2_error) < 0.5):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                            + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        
        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
        
```


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
```


```python
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:0.0% Speed(reviews/sec):0.0 #Correct:0 #Trained:1 Training Accuracy:0.0%
    Progress:10.4% Speed(reviews/sec):150.5 #Correct:1770 #Trained:2501 Training Accuracy:70.7%
    Progress:20.8% Speed(reviews/sec):150.5 #Correct:3719 #Trained:5001 Training Accuracy:74.3%
    Progress:31.2% Speed(reviews/sec):150.2 #Correct:5812 #Trained:7501 Training Accuracy:77.4%
    Progress:41.6% Speed(reviews/sec):150.3 #Correct:7932 #Trained:10001 Training Accuracy:79.3%
    Progress:52.0% Speed(reviews/sec):150.3 #Correct:10058 #Trained:12501 Training Accuracy:80.4%
    Progress:62.5% Speed(reviews/sec):150.2 #Correct:12192 #Trained:15001 Training Accuracy:81.2%
    Progress:72.9% Speed(reviews/sec):149.9 #Correct:14313 #Trained:17501 Training Accuracy:81.7%
    Progress:83.3% Speed(reviews/sec):149.9 #Correct:16486 #Trained:20001 Training Accuracy:82.4%
    Progress:93.7% Speed(reviews/sec):150.0 #Correct:18672 #Trained:22501 Training Accuracy:82.9%
    Progress:99.9% Speed(reviews/sec):150.1 #Correct:19999 #Trained:24000 Training Accuracy:83.3%


```python
# evaluate our model before training (just to show how horrible it is)
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):1621.% #Correct:858 #Tested:1000 Testing Accuracy:85.8%

# Analyzing Inefficiencies in our Network


```python
Image(filename='sentiment_network_sparse.png')
```




![png](/assets/img/neural_network/output_57_0.png)




```python
layer_0 = np.zeros(10)
```


```python
layer_0
```




    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])




```python
layer_0[4] = 1
layer_0[9] = 1
```


```python
layer_0
```




    array([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.])




```python
weights_0_1 = np.random.randn(10,5)
```


```python
layer_0.dot(weights_0_1)
```




    array([-0.10503756,  0.44222989,  0.24392938, -0.55961832,  0.21389503])




```python
indices = [4,9]
```


```python
layer_1 = np.zeros(5)
```


```python
for index in indices:
    layer_1 += (weights_0_1[index])
```


```python
layer_1
```




    array([-0.10503756,  0.44222989,  0.24392938, -0.55961832,  0.21389503])




```python
Image(filename='sentiment_network_sparse_2.png')
```




![png](/assets/img/neural_network/output_68_0.png)



# Project 5: Making our Network More Efficient


```python
import time
import sys

# Let's tweak our network from before to model these phenomena
class SentimentNetwork:
    def __init__(self, reviews,labels,hidden_nodes = 10, learning_rate = 0.1):
       
        np.random.seed(1)
    
        self.pre_process_data(reviews)
        
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)
        
        
    def pre_process_data(self,reviews):
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1,input_nodes))
        self.layer_1 = np.zeros((1,hidden_nodes))
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def update_input_layer(self,review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(" "):
            self.layer_0[0][self.word2index[word]] = 1

    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def train(self, training_reviews_raw, training_labels):
        
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))
        
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer

            # Hidden layer
#             layer_1 = self.layer_0.dot(self.weights_0_1)
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
            
            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            if(np.abs(layer_2_error) < 0.5):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
        
    
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                            + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        
        # Input Layer


        # Hidden layer
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
        
```


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
```


```python
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:99.9% Speed(reviews/sec):2564. #Correct:20113 #Trained:24000 Training Accuracy:83.8%


```python
# evaluate our model before training (just to show how horrible it is)
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):2995.% #Correct:853 #Tested:1000 Testing Accuracy:85.3%

# Further Noise Reduction


```python
Image(filename='sentiment_network_sparse_2.png')
```




![png](/assets/img/neural_network/output_75_0.png)




```python
# words most frequently seen in a review with a "POSITIVE" label
pos_neg_ratios.most_common()
```




    [('edie', 4.6913478822291435),
     ('paulie', 4.0775374439057197),
     ('felix', 3.1527360223636558),
     ('polanski', 2.8233610476132043),
     ('matthau', 2.8067217286092401),
     ('victoria', 2.6810215287142909),
     ('mildred', 2.6026896854443837),
     ('gandhi', 2.5389738710582761),
     ('flawless', 2.451005098112319),
     ('superbly', 2.2600254785752498),
     ('perfection', 2.1594842493533721),
     ('astaire', 2.1400661634962708),
     ('captures', 2.0386195471595809),
     ('voight', 2.0301704926730531),
     ('wonderfully', 2.0218960560332353),
     ('powell', 1.9783454248084671),
     ('brosnan', 1.9547990964725592),
     ('lily', 1.9203768470501485),
     ('bakshi', 1.9029851043382795),
     ('lincoln', 1.9014583864844796),
     ('refreshing', 1.8551812956655511),
     ('breathtaking', 1.8481124057791867),
     ('bourne', 1.8478489358790986),
     ('lemmon', 1.8458266904983307),
     ('delightful', 1.8002701588959635),
     ('flynn', 1.7996646487351682),
     ('andrews', 1.7764919970972666),
     ('homer', 1.7692866133759964),
     ('beautifully', 1.7626953362841438),
     ('soccer', 1.7578579175523736),
     ('elvira', 1.7397031072720019),
     ('underrated', 1.7197859696029656),
     ('gripping', 1.7165360479904674),
     ('superb', 1.7091514458966952),
     ('delight', 1.6714733033535532),
     ('welles', 1.6677068205580761),
     ('sadness', 1.663505133704376),
     ('sinatra', 1.6389967146756448),
     ('touching', 1.637217476541176),
     ('timeless', 1.62924053973028),
     ('macy', 1.6211339521972916),
     ('unforgettable', 1.6177367152487956),
     ('favorites', 1.6158688027643908),
     ('stewart', 1.6119987332957739),
     ('sullivan', 1.6094379124341003),
     ('extraordinary', 1.6094379124341003),
     ('hartley', 1.6094379124341003),
     ('brilliantly', 1.5950491749820008),
     ('friendship', 1.5677652160335325),
     ('wonderful', 1.5645425925262093),
     ('palma', 1.5553706911638245),
     ('magnificent', 1.54663701119507),
     ('finest', 1.5462590108125689),
     ('jackie', 1.5439233053234738),
     ('ritter', 1.5404450409471491),
     ('tremendous', 1.5184661342283736),
     ('freedom', 1.5091151908062312),
     ('fantastic', 1.5048433868558566),
     ('terrific', 1.5026699370083942),
     ('noir', 1.493925025312256),
     ('sidney', 1.493925025312256),
     ('outstanding', 1.4910053152089213),
     ('pleasantly', 1.4894785973551214),
     ('mann', 1.4894785973551214),
     ('nancy', 1.488077055429833),
     ('marie', 1.4825711915553104),
     ('marvelous', 1.4739999415389962),
     ('excellent', 1.4647538505723599),
     ('ruth', 1.4596256342054401),
     ('stanwyck', 1.4412101187160054),
     ('widmark', 1.4350845252893227),
     ('splendid', 1.4271163556401458),
     ('chan', 1.423108334242607),
     ('exceptional', 1.4201959127955721),
     ('tender', 1.410986973710262),
     ('gentle', 1.4078005663408544),
     ('poignant', 1.4022947024663317),
     ('gem', 1.3932148039644643),
     ('amazing', 1.3919815802404802),
     ('chilling', 1.3862943611198906),
     ('fisher', 1.3862943611198906),
     ('davies', 1.3862943611198906),
     ('captivating', 1.3862943611198906),
     ('darker', 1.3652409519220583),
     ('april', 1.3499267169490159),
     ('kelly', 1.3461743673304654),
     ('blake', 1.3418425985490567),
     ('overlooked', 1.329135947279942),
     ('ralph', 1.32818673031261),
     ('bette', 1.3156767939059373),
     ('hoffman', 1.3150668518315229),
     ('cole', 1.3121863889661687),
     ('shines', 1.3049487216659381),
     ('powerful', 1.2999662776313934),
     ('notch', 1.2950456896547455),
     ('remarkable', 1.2883688239495823),
     ('pitt', 1.286210902562908),
     ('winters', 1.2833463918674481),
     ('vivid', 1.2762934659055623),
     ('gritty', 1.2757524867200667),
     ('giallo', 1.2745029551317739),
     ('portrait', 1.2704625455947689),
     ('innocence', 1.2694300209805796),
     ('psychiatrist', 1.2685113254635072),
     ('favorite', 1.2668956297860055),
     ('ensemble', 1.2656663733312759),
     ('stunning', 1.2622417124499117),
     ('burns', 1.259880436264232),
     ('garbo', 1.258954938743289),
     ('barbara', 1.2580400255962119),
     ('philip', 1.2527629684953681),
     ('panic', 1.2527629684953681),
     ('holly', 1.2527629684953681),
     ('carol', 1.2481440226390734),
     ('perfect', 1.246742480713785),
     ('appreciated', 1.2462482874741743),
     ('favourite', 1.2411123512753928),
     ('journey', 1.2367626271489269),
     ('rural', 1.235471471385307),
     ('bond', 1.2321436812926323),
     ('builds', 1.2305398317106577),
     ('brilliant', 1.2287554137664785),
     ('brooklyn', 1.2286654169163074),
     ('von', 1.225175011976539),
     ('recommended', 1.2163953243244932),
     ('unfolds', 1.2163953243244932),
     ('daniel', 1.20215296760895),
     ('perfectly', 1.1971931173405572),
     ('crafted', 1.1962507582320256),
     ('prince', 1.1939224684724346),
     ('troubled', 1.192138346678933),
     ('consequences', 1.1865810616140668),
     ('haunting', 1.1814999484738773),
     ('cinderella', 1.180052620608284),
     ('alexander', 1.1759989522835299),
     ('emotions', 1.1753049094563641),
     ('boxing', 1.1735135968412274),
     ('subtle', 1.1734135017508081),
     ('curtis', 1.1649873576129823),
     ('rare', 1.1566438362402944),
     ('loved', 1.1563661500586044),
     ('daughters', 1.1526795099383853),
     ('courage', 1.1438688802562305),
     ('dentist', 1.1426722784621401),
     ('highly', 1.1420208631618658),
     ('nominated', 1.1409146683587992),
     ('tony', 1.1397491942285991),
     ('draws', 1.1325138403437911),
     ('everyday', 1.1306150197542835),
     ('contrast', 1.1284652518177909),
     ('cried', 1.1213405397456659),
     ('fabulous', 1.1210851445201684),
     ('ned', 1.120591195386885),
     ('fay', 1.120591195386885),
     ('emma', 1.1184149159642893),
     ('sensitive', 1.113318436057805),
     ('smooth', 1.1089750757036563),
     ('dramas', 1.1080910326226534),
     ('today', 1.1050431789984001),
     ('helps', 1.1023091505494358),
     ('inspiring', 1.0986122886681098),
     ('jimmy', 1.0937696641923216),
     ('awesome', 1.0931328229034842),
     ('unique', 1.0881409888008142),
     ('tragic', 1.0871835928444868),
     ('intense', 1.0870514662670339),
     ('stellar', 1.0857088838322018),
     ('rival', 1.0822184788924332),
     ('provides', 1.0797081340289569),
     ('depression', 1.0782034170369026),
     ('shy', 1.0775588794702773),
     ('carrie', 1.076139432816051),
     ('blend', 1.0753554265038423),
     ('hank', 1.0736109864626924),
     ('diana', 1.0726368022648489),
     ('adorable', 1.0726368022648489),
     ('unexpected', 1.0722255334949147),
     ('achievement', 1.0668635903535293),
     ('bettie', 1.0663514264498881),
     ('happiness', 1.0632729222228008),
     ('glorious', 1.0608719606852626),
     ('davis', 1.0541605260972757),
     ('terrifying', 1.0525211814678428),
     ('beauty', 1.050410186850232),
     ('ideal', 1.0479685558493548),
     ('fears', 1.0467872208035236),
     ('hong', 1.0438040521731147),
     ('seasons', 1.0433496099930604),
     ('fascinating', 1.0414538748281612),
     ('carries', 1.0345904299031787),
     ('satisfying', 1.0321225473992768),
     ('definite', 1.0319209141694374),
     ('touched', 1.0296194171811581),
     ('greatest', 1.0248947127715422),
     ('creates', 1.0241097613701886),
     ('aunt', 1.023388867430522),
     ('walter', 1.022328983918479),
     ('spectacular', 1.0198314108149955),
     ('portrayal', 1.0189810189761024),
     ('ann', 1.0127808528183286),
     ('enterprise', 1.0116009116784799),
     ('musicals', 1.0096648026516135),
     ('deeply', 1.0094845087721023),
     ('incredible', 1.0061677561461084),
     ('mature', 1.0060195018402847),
     ('triumph', 0.99682959435816731),
     ('margaret', 0.99682959435816731),
     ('navy', 0.99493385919326827),
     ('harry', 0.99176919305006062),
     ('lucas', 0.990398704027877),
     ('sweet', 0.98966110487955483),
     ('joey', 0.98794672078059009),
     ('oscar', 0.98721905111049713),
     ('balance', 0.98649499054740353),
     ('warm', 0.98485340331145166),
     ('ages', 0.98449898190068863),
     ('guilt', 0.98082925301172619),
     ('glover', 0.98082925301172619),
     ('carrey', 0.98082925301172619),
     ('learns', 0.97881108885548895),
     ('unusual', 0.97788374278196932),
     ('sons', 0.97777581552483595),
     ('complex', 0.97761897738147796),
     ('essence', 0.97753435711487369),
     ('brazil', 0.9769153536905899),
     ('widow', 0.97650959186720987),
     ('solid', 0.97537964824416146),
     ('beautiful', 0.97326301262841053),
     ('holmes', 0.97246100334120955),
     ('awe', 0.97186058302896583),
     ('vhs', 0.97116734209998934),
     ('eerie', 0.97116734209998934),
     ('lonely', 0.96873720724669754),
     ('grim', 0.96873720724669754),
     ('sport', 0.96825047080486615),
     ('debut', 0.96508089604358704),
     ('destiny', 0.96343751029985703),
     ('thrillers', 0.96281074750904794),
     ('tears', 0.95977584381389391),
     ('rose', 0.95664202739772253),
     ('feelings', 0.95551144502743635),
     ('ginger', 0.95551144502743635),
     ('winning', 0.95471810900804055),
     ('stanley', 0.95387344302319799),
     ('cox', 0.95343027882361187),
     ('paris', 0.95278479030472663),
     ('heart', 0.95238806924516806),
     ('hooked', 0.95155887071161305),
     ('comfortable', 0.94803943018873538),
     ('mgm', 0.94446160884085151),
     ('masterpiece', 0.94155039863339296),
     ('themes', 0.94118828349588235),
     ('danny', 0.93967118051821874),
     ('anime', 0.93378388932167222),
     ('perry', 0.93328830824272613),
     ('joy', 0.93301752567946861),
     ('lovable', 0.93081883243706487),
     ('mysteries', 0.92953595862417571),
     ('hal', 0.92953595862417571),
     ('louis', 0.92871325187271225),
     ('charming', 0.92520609553210742),
     ('urban', 0.92367083917177761),
     ('allows', 0.92183091224977043),
     ('impact', 0.91815814604895041),
     ('italy', 0.91629073187415511),
     ('gradually', 0.91629073187415511),
     ('lifestyle', 0.91629073187415511),
     ('spy', 0.91289514287301687),
     ('treat', 0.91193342650519937),
     ('subsequent', 0.91056005716517008),
     ('kennedy', 0.90981821736853763),
     ('loving', 0.90967549275543591),
     ('surprising', 0.90937028902958128),
     ('quiet', 0.90648673177753425),
     ('winter', 0.90624039602065365),
     ('reveals', 0.90490540964902977),
     ('raw', 0.90445627422715225),
     ('funniest', 0.90078654533818991),
     ('pleased', 0.89994159387262562),
     ('norman', 0.89994159387262562),
     ('thief', 0.89874642222324552),
     ('season', 0.89827222637147675),
     ('secrets', 0.89794159320595857),
     ('colorful', 0.89705936994626756),
     ('highest', 0.8967461358011849),
     ('compelling', 0.89462923509297576),
     ('danes', 0.89248008318043659),
     ('castle', 0.88967708335606499),
     ('kudos', 0.88889175768604067),
     ('great', 0.88810470901464589),
     ('baseball', 0.88730319500090271),
     ('subtitles', 0.88730319500090271),
     ('bleak', 0.88730319500090271),
     ('winner', 0.88643776872447388),
     ('tragedy', 0.88563699078315261),
     ('todd', 0.88551907320740142),
     ('nicely', 0.87924946019380601),
     ('arthur', 0.87546873735389985),
     ('essential', 0.87373111745535925),
     ('gorgeous', 0.8731725250935497),
     ('fonda', 0.87294029100054127),
     ('eastwood', 0.87139541196626402),
     ('focuses', 0.87082835779739776),
     ('enjoyed', 0.87070195951624607),
     ('natural', 0.86997924506912838),
     ('intensity', 0.86835126958503595),
     ('witty', 0.86824103423244681),
     ('rob', 0.8642954367557748),
     ('worlds', 0.86377269759070874),
     ('health', 0.86113891179907498),
     ('magical', 0.85953791528170564),
     ('deeper', 0.85802182375017932),
     ('lucy', 0.85618680780444956),
     ('moving', 0.85566611005772031),
     ('lovely', 0.85290640004681306),
     ('purple', 0.8513711857748395),
     ('memorable', 0.84801189112086062),
     ('sings', 0.84729786038720367),
     ('craig', 0.84342938360928321),
     ('modesty', 0.84342938360928321),
     ('relate', 0.84326559685926517),
     ('episodes', 0.84223712084137292),
     ('strong', 0.84167135777060931),
     ('smith', 0.83959811108590054),
     ('tear', 0.83704136022001441),
     ('apartment', 0.83333115290549531),
     ('princess', 0.83290912293510388),
     ('disagree', 0.83290912293510388),
     ('kung', 0.83173334384609199),
     ('adventure', 0.83150561393278388),
     ('columbo', 0.82667857318446791),
     ('jake', 0.82667857318446791),
     ('adds', 0.82485652591452319),
     ('hart', 0.82472353834866463),
     ('strength', 0.82417544296634937),
     ('realizes', 0.82360006895738058),
     ('dave', 0.8232003088081431),
     ('childhood', 0.82208086393583857),
     ('forbidden', 0.81989888619908913),
     ('tight', 0.81883539572344199),
     ('surreal', 0.8178506590609026),
     ('manager', 0.81770990320170756),
     ('dancer', 0.81574950265227764),
     ('studios', 0.81093021621632877),
     ('con', 0.81093021621632877),
     ('miike', 0.80821651034473263),
     ('realistic', 0.80807714723392232),
     ('explicit', 0.80792269515237358),
     ('kurt', 0.8060875917405409),
     ('traditional', 0.80535917116687328),
     ('deals', 0.80535917116687328),
     ('holds', 0.80493858654806194),
     ('carl', 0.80437281567016972),
     ('touches', 0.80396154690023547),
     ('gene', 0.80314807577427383),
     ('albert', 0.8027669055771679),
     ('abc', 0.80234647252493729),
     ('cry', 0.80011930011211307),
     ('sides', 0.7995275841185171),
     ('develops', 0.79850769621777162),
     ('eyre', 0.79850769621777162),
     ('dances', 0.79694397424158891),
     ('oscars', 0.79633141679517616),
     ('legendary', 0.79600456599965308),
     ('hearted', 0.79492987486988764),
     ('importance', 0.79492987486988764),
     ('portraying', 0.79356592830699269),
     ('impressed', 0.79258107754813223),
     ('waters', 0.79112758892014912),
     ('empire', 0.79078565012386137),
     ('edge', 0.789774016249017),
     ('jean', 0.78845736036427028),
     ('environment', 0.78845736036427028),
     ('sentimental', 0.7864791203521645),
     ('captured', 0.78623760362595729),
     ('styles', 0.78592891401091158),
     ('daring', 0.78592891401091158),
     ('frank', 0.78275933924963248),
     ('tense', 0.78275933924963248),
     ('backgrounds', 0.78275933924963248),
     ('matches', 0.78275933924963248),
     ('gothic', 0.78209466657644144),
     ('sharp', 0.7814397877056235),
     ('achieved', 0.78015855754957497),
     ('court', 0.77947526404844247),
     ('steals', 0.7789140023173704),
     ('rules', 0.77844476107184035),
     ('colors', 0.77684619943659217),
     ('reunion', 0.77318988823348167),
     ('covers', 0.77139937745969345),
     ('tale', 0.77010822169607374),
     ('rain', 0.7683706017975328),
     ('denzel', 0.76804848873306297),
     ('stays', 0.76787072675588186),
     ('blob', 0.76725515271366718),
     ('maria', 0.76214005204689672),
     ('conventional', 0.76214005204689672),
     ('fresh', 0.76158434211317383),
     ('midnight', 0.76096977689870637),
     ('landscape', 0.75852993982279704),
     ('animated', 0.75768570169751648),
     ('titanic', 0.75666058628227129),
     ('sunday', 0.75666058628227129),
     ('spring', 0.7537718023763802),
     ('cagney', 0.7537718023763802),
     ('enjoyable', 0.75246375771636476),
     ('immensely', 0.75198768058287868),
     ('sir', 0.7507762933965817),
     ('nevertheless', 0.75067102469813185),
     ('driven', 0.74994477895307854),
     ('performances', 0.74883252516063137),
     ('memories', 0.74721440183022114),
     ('nowadays', 0.74721440183022114),
     ('simple', 0.74641420974143258),
     ('golden', 0.74533293373051557),
     ('leslie', 0.74533293373051557),
     ('lovers', 0.74497224842453125),
     ('relationship', 0.74484232345601786),
     ('supporting', 0.74357803418683721),
     ('che', 0.74262723782331497),
     ('packed', 0.7410032017375805),
     ('trek', 0.74021469141793106),
     ('provoking', 0.73840377214806618),
     ('strikes', 0.73759894313077912),
     ('depiction', 0.73682224406260699),
     ('emotional', 0.73678211645681524),
     ('secretary', 0.7366322924996842),
     ('influenced', 0.73511137965897755),
     ('florida', 0.73511137965897755),
     ('germany', 0.73288750920945944),
     ('brings', 0.73142936713096229),
     ('lewis', 0.73129894652432159),
     ('elderly', 0.73088750854279239),
     ('owner', 0.72743625403857748),
     ('streets', 0.72666987259858895),
     ('henry', 0.72642196944481741),
     ('portrays', 0.72593700338293632),
     ('bears', 0.7252354951114458),
     ('china', 0.72489587887452556),
     ('anger', 0.72439972406404984),
     ('society', 0.72433010799663333),
     ('available', 0.72415741730250549),
     ('best', 0.72347034060446314),
     ('bugs', 0.72270598280148979),
     ('magic', 0.71878961117328299),
     ('delivers', 0.71846498854423513),
     ('verhoeven', 0.71846498854423513),
     ('jim', 0.71783979315031676),
     ('donald', 0.71667767797013937),
     ('endearing', 0.71465338578090898),
     ('relationships', 0.71393795022901896),
     ('greatly', 0.71256526641704687),
     ('charlie', 0.71024161391924534),
     ('brad', 0.71024161391924534),
     ('simon', 0.70967648251115578),
     ('effectively', 0.70914752190638641),
     ('march', 0.70774597998109789),
     ('atmosphere', 0.70744773070214162),
     ('influence', 0.70733181555190172),
     ('genius', 0.706392407309966),
     ('emotionally', 0.70556970055850243),
     ('ken', 0.70526854109229009),
     ('identity', 0.70484322032313651),
     ('sophisticated', 0.70470800296102132),
     ('dan', 0.70457587638356811),
     ('andrew', 0.70329955202396321),
     ('india', 0.70144598337464037),
     ('roy', 0.69970458110610434),
     ('surprisingly', 0.6995780708902356),
     ('sky', 0.69780919366575667),
     ('romantic', 0.69664981111114743),
     ('match', 0.69566924999265523),
     ('meets', 0.69314718055994529),
     ('cowboy', 0.69314718055994529),
     ('wave', 0.69314718055994529),
     ('bitter', 0.69314718055994529),
     ('patient', 0.69314718055994529),
     ('stylish', 0.69314718055994529),
     ('britain', 0.69314718055994529),
     ('affected', 0.69314718055994529),
     ('beatty', 0.69314718055994529),
     ('love', 0.69198533541937324),
     ('paul', 0.68980827929443067),
     ('andy', 0.68846333124751902),
     ('performance', 0.68797386327972465),
     ('patrick', 0.68645819240914863),
     ('unlike', 0.68546468438792907),
     ('brooks', 0.68433655087779044),
     ('refuses', 0.68348526964820844),
     ('award', 0.6824518914431974),
     ('complaint', 0.6824518914431974),
     ('ride', 0.68229716453587952),
     ('dawson', 0.68171848473632257),
     ('luke', 0.68158635815886937),
     ('wells', 0.68087708796813096),
     ('france', 0.6804081547825156),
     ('sports', 0.68007509899259255),
     ('handsome', 0.68007509899259255),
     ('directs', 0.67875844310784572),
     ('rebel', 0.67875844310784572),
     ('greater', 0.67605274720064523),
     ('dreams', 0.67599410133369586),
     ('effective', 0.67565402311242806),
     ('interpretation', 0.67479804189174875),
     ('works', 0.67445504754779284),
     ('brando', 0.67445504754779284),
     ('noble', 0.6737290947028437),
     ('paced', 0.67314651385327573),
     ('le', 0.67067432470788668),
     ('master', 0.67015766233524654),
     ('h', 0.6696166831497512),
     ('rings', 0.66904962898088483),
     ('easy', 0.66895995494594152),
     ('city', 0.66820823221269321),
     ('sunshine', 0.66782937257565544),
     ('succeeds', 0.66647893347778397),
     ('relations', 0.664159643686693),
     ('england', 0.66387679825983203),
     ('glimpse', 0.66329421741026418),
     ('aired', 0.66268797307523675),
     ('sees', 0.66263163663399482),
     ('both', 0.66248336767382998),
     ('definitely', 0.66199789483898808),
     ('imaginative', 0.66139848224536502),
     ('appreciate', 0.66083893732728749),
     ('tricks', 0.66071190480679143),
     ('striking', 0.66071190480679143),
     ('carefully', 0.65999497324304479),
     ('complicated', 0.65981076029235353),
     ('perspective', 0.65962448852130173),
     ('trilogy', 0.65877953705573755),
     ('future', 0.65834665141052828),
     ('lion', 0.65742909795786608),
     ('douglas', 0.65540685257709819),
     ('victor', 0.65540685257709819),
     ('inspired', 0.65459851044271034),
     ('marriage', 0.65392646740666405),
     ('demands', 0.65392646740666405),
     ('father', 0.65172321672194655),
     ('page', 0.65123628494430852),
     ('instant', 0.65058756614114943),
     ('era', 0.6495567444850836),
     ('ruthless', 0.64934455790155243),
     ('saga', 0.64934455790155243),
     ('joan', 0.64891392558311978),
     ('joseph', 0.64841128671855386),
     ('workers', 0.64829661439459352),
     ('fantasy', 0.64726757480925168),
     ('distant', 0.64551913157069074),
     ('accomplished', 0.64551913157069074),
     ('manhattan', 0.64435701639051324),
     ('personal', 0.64355023942057321),
     ('meeting', 0.64313675998528386),
     ('individual', 0.64313675998528386),
     ('pushing', 0.64313675998528386),
     ('pleasant', 0.64250344774119039),
     ('brave', 0.64185388617239469),
     ('william', 0.64083139119578469),
     ('hudson', 0.64077919504262937),
     ('friendly', 0.63949446706762514),
     ('eccentric', 0.63907995928966954),
     ('awards', 0.63875310849414646),
     ('jack', 0.63838309514997038),
     ('seeking', 0.63808740337691783),
     ('divorce', 0.63757732940513456),
     ('colonel', 0.63757732940513456),
     ('jane', 0.63443957973316734),
     ('keeping', 0.63414883979798953),
     ('gives', 0.63383568159497883),
     ('ted', 0.63342794585832296),
     ('animation', 0.63208692379869902),
     ('progress', 0.6317782341836532),
     ('larger', 0.63127177684185776),
     ('concert', 0.63127177684185776),
     ('nation', 0.6296337748376194),
     ('albeit', 0.62739580299716491),
     ('adapted', 0.62613647027698516),
     ('discovers', 0.62542900650499444),
     ('classic', 0.62504956428050518),
     ('segment', 0.62335141862440335),
     ('morgan', 0.62303761437291871),
     ('mouse', 0.62294292188669675),
     ('impressive', 0.62211140744319349),
     ('artist', 0.62168821657780038),
     ('ultimate', 0.62168821657780038),
     ('griffith', 0.62117368093485603),
     ('drew', 0.62082651898031915),
     ('emily', 0.62082651898031915),
     ('moved', 0.6197197120051281),
     ('families', 0.61903920840622351),
     ('profound', 0.61903920840622351),
     ('innocent', 0.61851219917136446),
     ('versions', 0.61730910416844087),
     ('eddie', 0.61691981517206107),
     ('criticism', 0.61651395453902935),
     ('nature', 0.61594514653194088),
     ('recognized', 0.61518563909023349),
     ('sexuality', 0.61467556511845012),
     ('contract', 0.61400986000122149),
     ('brian', 0.61344043794920278),
     ('remembered', 0.6131044728864089),
     ('determined', 0.6123858239154869),
     ('offers', 0.61207935747116349),
     ('pleasure', 0.61195702582993206),
     ('washington', 0.61180154110599294),
     ('images', 0.61159731359583758),
     ('games', 0.61067095873570676),
     ('academy', 0.60872983874736208),
     ('fashioned', 0.60798937221963845),
     ('melodrama', 0.60749173598145145),
     ('rough', 0.60613580357031549),
     ('charismatic', 0.60613580357031549),
     ('peoples', 0.60613580357031549),
     ('dealing', 0.60517840761398811),
     ('fine', 0.60496962268013299),
     ('tap', 0.60391604683200273),
     ('trio', 0.60157998703445481),
     ('russell', 0.60120968523425966),
     ('figures', 0.60077386042893011),
     ('ward', 0.60005675749393339),
     ('shine', 0.59911823091166894),
     ('brady', 0.59911823091166894),
     ('job', 0.59845562125168661),
     ('satisfied', 0.59652034487087369),
     ('river', 0.59637962862495086),
     ('brown', 0.595773016534769),
     ('believable', 0.59566072133302495),
     ('always', 0.59470710774669278),
     ('bound', 0.59470710774669278),
     ('hall', 0.5933967777928858),
     ('cook', 0.5916777203950857),
     ('claire', 0.59136448625000293),
     ('broadway', 0.59033768669372433),
     ('anna', 0.58778666490211906),
     ('peace', 0.58628403501758408),
     ('visually', 0.58539431926349916),
     ('morality', 0.58525821854876026),
     ('falk', 0.58525821854876026),
     ('growing', 0.58466653756587539),
     ('experiences', 0.58314628534561685),
     ('stood', 0.58314628534561685),
     ('touch', 0.58122926435596001),
     ('lives', 0.5810976767513224),
     ('kubrick', 0.58066919713325493),
     ('timing', 0.58047401805583243),
     ('expressions', 0.57981849525294216),
     ('struggles', 0.57981849525294216),
     ('authentic', 0.57848427223980559),
     ('helen', 0.57763429343810091),
     ('pre', 0.57700753064729182),
     ('quirky', 0.5753641449035618),
     ('young', 0.57531672344534313),
     ('inner', 0.57454143815209846),
     ('mexico', 0.57443087372056334),
     ('clint', 0.57380042292737909),
     ('sisters', 0.57286101468544337),
     ('realism', 0.57226528899949558),
     ('french', 0.5720692490067093),
     ('personalities', 0.5720692490067093),
     ('surprises', 0.57113222999698177),
     ('adventures', 0.57113222999698177),
     ('overcome', 0.5697681593994407),
     ('timothy', 0.56953322459276867),
     ('tales', 0.56909453188996639),
     ('war', 0.56843317302781682),
     ('civil', 0.5679840376059393),
     ('countries', 0.56737779327091187),
     ('streep', 0.56710645966458029),
     ('tradition', 0.56685345523565323),
     ('oliver', 0.56673325570428668),
     ('australia', 0.56580775818334383),
     ('understanding', 0.56531380905006046),
     ('players', 0.56509525370004821),
     ('knowing', 0.56489284503626647),
     ('rogers', 0.56421349718405212),
     ('suspenseful', 0.56368911332305849),
     ('variety', 0.56368911332305849),
     ('true', 0.56281525180810066),
     ('jr', 0.56220982311246936),
     ('psychological', 0.56108745854687891),
     ('sent', 0.55961578793542266),
     ('grand', 0.55961578793542266),
     ('branagh', 0.55961578793542266),
     ('reminiscent', 0.55961578793542266),
     ('performing', 0.55961578793542266),
     ('wealth', 0.55961578793542266),
     ('overwhelming', 0.55961578793542266),
     ('odds', 0.55961578793542266),
     ('brothers', 0.55891181043362848),
     ('howard', 0.55811089675600245),
     ('david', 0.55693122256475369),
     ('generation', 0.55628799784274796),
     ('grow', 0.55612538299565417),
     ('survival', 0.55594605904646033),
     ('mainstream', 0.55574731115750231),
     ('dick', 0.55431073570572953),
     ('charm', 0.55288175575407861),
     ('kirk', 0.55278982286502287),
     ('twists', 0.55244729845681018),
     ('gangster', 0.55206858230003986),
     ('jeff', 0.55179306225421365),
     ('family', 0.55116244510065526),
     ('tend', 0.55053307336110335),
     ('thanks', 0.55049088015842218),
     ('world', 0.54744234723432639),
     ('sutherland', 0.54743536937855164),
     ('life', 0.54695514434959924),
     ('disc', 0.54654370636806993),
     ('bug', 0.54654370636806993),
     ('tribute', 0.5455111817538808),
     ('europe', 0.54522705048332309),
     ('sacrifice', 0.54430155296238014),
     ('color', 0.54405127139431109),
     ('superior', 0.54333490233128523),
     ('york', 0.54318235866536513),
     ('pulls', 0.54266622962164945),
     ('jackson', 0.54232429082536171),
     ('hearts', 0.54232429082536171),
     ('enjoy', 0.54124285135906114),
     ('redemption', 0.54056759296472823),
     ('madness', 0.540384426007535),
     ('stands', 0.5389965007326869),
     ('trial', 0.5389965007326869),
     ('greek', 0.5389965007326869),
     ('hamilton', 0.5389965007326869),
     ('each', 0.5388212312554177),
     ('faithful', 0.53773307668591508),
     ('received', 0.5372768098531604),
     ('documentaries', 0.53714293208336406),
     ('jealous', 0.53714293208336406),
     ('different', 0.53709860682460819),
     ('describes', 0.53680111016925136),
     ('shorts', 0.53596159703753288),
     ('brilliance', 0.53551823635636209),
     ('mountains', 0.53492317534505118),
     ('share', 0.53408248593025787),
     ('dealt', 0.53408248593025787),
     ('providing', 0.53329847961804933),
     ('explore', 0.53329847961804933),
     ('series', 0.5325809226575603),
     ('fellow', 0.5323318289869543),
     ('loves', 0.53062825106217038),
     ('revolution', 0.53062825106217038),
     ('olivier', 0.53062825106217038),
     ('roman', 0.53062825106217038),
     ('century', 0.53002783074992665),
     ('musical', 0.52966871156747064),
     ('heroic', 0.52925932545482868),
     ('approach', 0.52806743020049673),
     ('ironically', 0.52806743020049673),
     ('temple', 0.52806743020049673),
     ('moves', 0.5279372642387119),
     ('gift', 0.52702030968597136),
     ('julie', 0.52609309589677911),
     ('tells', 0.52415107836314001),
     ('radio', 0.52394671172868779),
     ('uncle', 0.52354439617376536),
     ('union', 0.52324814376454787),
     ('deep', 0.52309571635780505),
     ('reminds', 0.52157841554225237),
     ('famous', 0.52118841080153722),
     ('jazz', 0.52053443789295151),
     ('dennis', 0.51987545928590861),
     ('epic', 0.51919387343650736),
     ('adult', 0.519167695083386),
     ('shows', 0.51915322220375304),
     ('performed', 0.5191244265806858),
     ('demons', 0.5191244265806858),
     ('discovered', 0.51879379341516751),
     ('eric', 0.51879379341516751),
     ('youth', 0.5185626062681431),
     ('human', 0.51851411224987087),
     ('tarzan', 0.51813827061227724),
     ('ourselves', 0.51794309153485463),
     ('wwii', 0.51758240622887042),
     ('passion', 0.5162164724008671),
     ('desire', 0.51607497965213445),
     ('pays', 0.51581316527702981),
     ('dirty', 0.51557622652458857),
     ('fox', 0.51557622652458857),
     ('sympathetic', 0.51546600332249293),
     ('symbolism', 0.51546600332249293),
     ('attitude', 0.51530993621331933),
     ('appearances', 0.51466440007315639),
     ('jeremy', 0.51466440007315639),
     ('fun', 0.51439068993048687),
     ('south', 0.51420972175023116),
     ('arrives', 0.51409894911095988),
     ('present', 0.51341965894303732),
     ('com', 0.51326167856387173),
     ('smile', 0.51265880484765169),
     ('alan', 0.51082562376599072),
     ('ring', 0.51082562376599072),
     ('visit', 0.51082562376599072),
     ('fits', 0.51082562376599072),
     ('provided', 0.51082562376599072),
     ('carter', 0.51082562376599072),
     ('aging', 0.51082562376599072),
     ('countryside', 0.51082562376599072),
     ('begins', 0.51015650363396647),
     ('success', 0.50900578704900468),
     ('japan', 0.50900578704900468),
     ('accurate', 0.50895471583017893),
     ('proud', 0.50800474742434931),
     ('daily', 0.5075946031845443),
     ('karloff', 0.50724780241810674),
     ('atmospheric', 0.50724780241810674),
     ('recently', 0.50714914903668207),
     ('fu', 0.50704490092608467),
     ('horrors', 0.50656122497953315),
     ('finding', 0.50637127341661037),
     ('lust', 0.5059356384717989),
     ('hitchcock', 0.50574947073413001),
     ('among', 0.50334004951332734),
     ('viewing', 0.50302139827440906),
     ('investigation', 0.50262885656181222),
     ('shining', 0.50262885656181222),
     ('duo', 0.5020919437972361),
     ('cameron', 0.5020919437972361),
     ('finds', 0.50128303100539795),
     ('contemporary', 0.50077528791248915),
     ('genuine', 0.50046283673044401),
     ('frightening', 0.49995595152908684),
     ('plays', 0.49975983848890226),
     ('age', 0.49941323171424595),
     ('position', 0.49899116611898781),
     ('continues', 0.49863035067217237),
     ('roles', 0.49839716550752178),
     ('james', 0.49837216269470402),
     ('individuals', 0.49824684155913052),
     ('brought', 0.49783842823917956),
     ('hilarious', 0.49714551986191058),
     ('brutal', 0.49681488669639234),
     ('appropriate', 0.49643688631389105),
     ('dance', 0.49581998314812048),
     ('league', 0.49578774640145024),
     ('helping', 0.49578774640145024),
     ('answers', 0.49578774640145024),
     ('stunts', 0.49561620510246196),
     ('traveling', 0.49532143723002542),
     ('thoroughly', 0.49414593456733524),
     ('depicted', 0.49317068852726992),
     ('combination', 0.49247648509779424),
     ('honor', 0.49247648509779424),
     ('differences', 0.49247648509779424),
     ('fully', 0.49213349075383811),
     ('tracy', 0.49159426183810306),
     ('battles', 0.49140753790888908),
     ('possibility', 0.49112055268665822),
     ('romance', 0.4901589869574316),
     ('initially', 0.49002249613622745),
     ('happy', 0.4898997500608791),
     ('crime', 0.48977221456815834),
     ('singing', 0.4893852925281213),
     ('especially', 0.48901267837860624),
     ('shakespeare', 0.48754793889664511),
     ('hugh', 0.48729512635579658),
     ('detail', 0.48609484250827351),
     ('julia', 0.48550781578170082),
     ('san', 0.48550781578170082),
     ('guide', 0.48550781578170082),
     ('desperation', 0.48550781578170082),
     ('companion', 0.48550781578170082),
     ('strongly', 0.48460242866688824),
     ('necessary', 0.48302334245403883),
     ('humanity', 0.48265474679929443),
     ('drama', 0.48221998493060503),
     ('nonetheless', 0.48183808689273838),
     ('intrigue', 0.48183808689273838),
     ('warming', 0.48183808689273838),
     ('cuba', 0.48183808689273838),
     ('planned', 0.47957308026188628),
     ('pictures', 0.47929937011921681),
     ('broadcast', 0.47849024312305422),
     ('nine', 0.47803580094299974),
     ('settings', 0.47743860773325364),
     ('history', 0.47732966933780852),
     ('ordinary', 0.47725880012690741),
     ('trade', 0.47692407209030935),
     ('official', 0.47608267532211779),
     ('primary', 0.47608267532211779),
     ('episode', 0.47529620261150429),
     ('role', 0.47520268270188676),
     ('spirit', 0.47477690799839323),
     ('grey', 0.47409361449726067),
     ('ways', 0.47323464982718205),
     ('cup', 0.47260441094579297),
     ('piano', 0.47260441094579297),
     ('familiar', 0.47241617565111949),
     ('sinister', 0.47198579044972683),
     ('reveal', 0.47171449364936496),
     ('max', 0.47150852042515579),
     ('dated', 0.47121648567094482),
     ('losing', 0.47000362924573563),
     ('discovery', 0.47000362924573563),
     ('vicious', 0.47000362924573563),
     ('genuinely', 0.46871413841586385),
     ('hatred', 0.46734051182625186),
     ('mistaken', 0.46702300110759781),
     ('dream', 0.46608972992459924),
     ('challenge', 0.46608972992459924),
     ('crisis', 0.46575733836428446),
     ('photographed', 0.46488852857896512),
     ('critics', 0.46430560813109778),
     ('bird', 0.46430560813109778),
     ('machines', 0.46430560813109778),
     ('born', 0.46411383518967209),
     ('detective', 0.4636633473511525),
     ('higher', 0.46328467899699055),
     ('remains', 0.46262352194811296),
     ('inevitable', 0.46262352194811296),
     ('soviet', 0.4618180446592961),
     ('ryan', 0.46134556650262099),
     ('african', 0.46112595521371813),
     ('smaller', 0.46081520319132935),
     ('techniques', 0.46052488529119184),
     ('information', 0.46034171833399862),
     ('deserved', 0.45999798712841444),
     ('lynch', 0.45953232937844013),
     ('spielberg', 0.45953232937844013),
     ('cynical', 0.45953232937844013),
     ('tour', 0.45953232937844013),
     ('francisco', 0.45953232937844013),
     ('struggle', 0.45911782160048453),
     ('language', 0.45902121257712653),
     ('visual', 0.45823514408822852),
     ('warner', 0.45724137763188427),
     ('social', 0.45720078250735313),
     ('reality', 0.45719346885019546),
     ('hidden', 0.45675840249571492),
     ('breaking', 0.45601738727099561),
     ('sometimes', 0.45563021171182794),
     ('modern', 0.45500247579345005),
     ('surfing', 0.45425527227759638),
     ('popular', 0.45410691533051023),
     ('surprised', 0.4534409399850382),
     ('follows', 0.45245361754408348),
     ('keeps', 0.45234869400701483),
     ('john', 0.4520909494482197),
     ('mixed', 0.45198512374305722),
     ('defeat', 0.45198512374305722),
     ('justice', 0.45142724367280018),
     ('treasure', 0.45083371313801535),
     ('presents', 0.44973793178615257),
     ('years', 0.44919197032104968),
     ('chief', 0.44895022004790319),
     ('shadows', 0.44802472252696035),
     ('closely', 0.44701411102103689),
     ('segments', 0.44701411102103689),
     ('lose', 0.44658335503763702),
     ('caine', 0.44628710262841953),
     ('caught', 0.44610275383999071),
     ('hamlet', 0.44558510189758965),
     ('chinese', 0.44507424620321018),
     ('welcome', 0.44438052435783792),
     ('birth', 0.44368632092836219),
     ('represents', 0.44320543609101143),
     ('puts', 0.44279106572085081),
     ('visuals', 0.44183275227903923),
     ('fame', 0.44183275227903923),
     ('closer', 0.44183275227903923),
     ('web', 0.44183275227903923),
     ('criminal', 0.4412745608048752),
     ('minor', 0.4409224199448939),
     ('jon', 0.44086703515908027),
     ('liked', 0.44074991514020723),
     ('restaurant', 0.44031183943833246),
     ('de', 0.43983275161237217),
     ('flaws', 0.43983275161237217),
     ('searching', 0.4393666597838457),
     ('rap', 0.43891304217570443),
     ('light', 0.43884433018199892),
     ('elizabeth', 0.43872232986464677),
     ('marry', 0.43861731542506488),
     ('learned', 0.43825493093115531),
     ('controversial', 0.43825493093115531),
     ('oz', 0.43825493093115531),
     ('slowly', 0.43785660389939979),
     ('comedic', 0.43721380642274466),
     ('wayne', 0.43721380642274466),
     ('thrilling', 0.43721380642274466),
     ('bridge', 0.43721380642274466),
     ('married', 0.43658501682196887),
     ('nazi', 0.4361020775700542),
     ('murder', 0.4353180712578455),
     ('physical', 0.4353180712578455),
     ('johnny', 0.43483971678806865),
     ('michelle', 0.43445264498141672),
     ('wallace', 0.43403848055222038),
     ('comedies', 0.43395706390247063),
     ('silent', 0.43395706390247063),
     ('played', 0.43387244114515305),
     ('international', 0.43363598507486073),
     ('vision', 0.43286408229627887),
     ('intelligent', 0.43196704885367099),
     ('shop', 0.43078291609245434),
     ('also', 0.43036720209769169),
     ('levels', 0.4302451371066513),
     ('miss', 0.43006426712153217),
     ('movement', 0.4295626596872249),
     ...]




```python
# words most frequently seen in a review with a "NEGATIVE" label
list(reversed(pos_neg_ratios.most_common()))[0:30]
```




    [('boll', -4.0778152602708904),
     ('uwe', -3.9218753018711578),
     ('seagal', -3.3202501058581921),
     ('unwatchable', -3.0269848170580955),
     ('stinker', -2.9876839403711624),
     ('mst', -2.7753833211707968),
     ('incoherent', -2.7641396677532537),
     ('unfunny', -2.5545257844967644),
     ('waste', -2.4907515123361046),
     ('blah', -2.4475792789485005),
     ('horrid', -2.3715779644809971),
     ('pointless', -2.3451073877136341),
     ('atrocious', -2.3187369339642556),
     ('redeeming', -2.2667790015910296),
     ('prom', -2.2601040980178784),
     ('drivel', -2.2476029585766928),
     ('lousy', -2.2118080125207054),
     ('worst', -2.1930856334332267),
     ('laughable', -2.172468615469592),
     ('awful', -2.1385076866397488),
     ('poorly', -2.1326133844207011),
     ('wasting', -2.1178155545614512),
     ('remotely', -2.111046881095167),
     ('existent', -2.0024805005437076),
     ('boredom', -1.9241486572738005),
     ('miserably', -1.9216610938019989),
     ('sucks', -1.9166645809588516),
     ('uninspired', -1.9131499212248517),
     ('lame', -1.9117232884159072),
     ('insult', -1.9085323769376259)]




```python
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
output_notebook()
```



    <div class="bk-root">
        <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="67a2b150-b2f3-44b7-a401-91d2f2e64a0b">Loading BokehJS ...</span>
    </div>





```python
hist, edges = np.histogram(list(map(lambda x:x[1],pos_neg_ratios.most_common())), density=True, bins=100, normed=True)

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="Word Positive/Negative Affinity Distribution")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
```




    <div class="bk-root">
        <div class="bk-plotdiv" id="09af843a-8ba0-4f13-a758-59453c5e2213"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        document.getElementById("09af843a-8ba0-4f13-a758-59453c5e2213").textContent = "BokehJS successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("09af843a-8ba0-4f13-a758-59453c5e2213");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid '09af843a-8ba0-4f13-a758-59453c5e2213' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"932bbd7d-9cfa-4f24-a68d-65a2f3b9d077":{"roots":{"references":[{"attributes":{"plot":{"id":"7ae81053-1f24-48a9-85e4-b30704bbc24b","subtype":"Figure","type":"Plot"},"ticker":{"id":"fceb939b-a091-488b-89d4-4a1289254b2b","type":"BasicTicker"}},"id":"bd6c6269-c94e-423b-a26b-41c0a1d3da33","type":"Grid"},{"attributes":{"formatter":{"id":"56803c7f-e6cd-4496-8898-175419c75307","type":"BasicTickFormatter"},"plot":{"id":"7ae81053-1f24-48a9-85e4-b30704bbc24b","subtype":"Figure","type":"Plot"},"ticker":{"id":"a401f008-4871-4894-80ee-d6b81cbac492","type":"BasicTicker"}},"id":"873f13d5-ad53-41cc-9e68-558b3625adb1","type":"LinearAxis"},{"attributes":{},"id":"771947ef-d589-49ae-98f5-6ef45a8c1024","type":"BasicTickFormatter"},{"attributes":{},"id":"a401f008-4871-4894-80ee-d6b81cbac492","type":"BasicTicker"},{"attributes":{"dimension":1,"plot":{"id":"7ae81053-1f24-48a9-85e4-b30704bbc24b","subtype":"Figure","type":"Plot"},"ticker":{"id":"a401f008-4871-4894-80ee-d6b81cbac492","type":"BasicTicker"}},"id":"98985a39-f4f7-447e-90da-8df0b5326fe4","type":"Grid"},{"attributes":{"data_source":{"id":"7ffb53d7-2954-42ee-8006-9db7d370a58a","type":"ColumnDataSource"},"glyph":{"id":"9ccfdd0e-2895-4f93-b5bb-b0f46e0d1dba","type":"Quad"},"hover_glyph":null,"nonselection_glyph":{"id":"c19a3f86-aa36-40c1-b111-a8f7593ec01f","type":"Quad"},"selection_glyph":null},"id":"458f8b66-92f2-4945-8953-5ff8544870b6","type":"GlyphRenderer"},{"attributes":{"plot":{"id":"7ae81053-1f24-48a9-85e4-b30704bbc24b","subtype":"Figure","type":"Plot"}},"id":"29bf3458-2214-4e1c-8371-cf99a3e3d1ee","type":"PanTool"},{"attributes":{"plot":null,"text":"Word Positive/Negative Affinity Distribution"},"id":"5e6aebc1-300c-4468-bae9-518ac28ca463","type":"Title"},{"attributes":{"plot":{"id":"7ae81053-1f24-48a9-85e4-b30704bbc24b","subtype":"Figure","type":"Plot"}},"id":"08797199-4344-44d6-a585-ecd6eec0e0e5","type":"WheelZoomTool"},{"attributes":{"plot":{"id":"7ae81053-1f24-48a9-85e4-b30704bbc24b","subtype":"Figure","type":"Plot"}},"id":"c922a12f-3b41-4cfe-ad44-30cb288bd738","type":"ResetTool"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"right":{"field":"right"},"top":{"field":"top"}},"id":"c19a3f86-aa36-40c1-b111-a8f7593ec01f","type":"Quad"},{"attributes":{"plot":{"id":"7ae81053-1f24-48a9-85e4-b30704bbc24b","subtype":"Figure","type":"Plot"}},"id":"c8c8f3b7-92a6-48de-a5a7-a7c8745d236d","type":"SaveTool"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#1f77b4"},"left":{"field":"left"},"line_color":{"value":"#555555"},"right":{"field":"right"},"top":{"field":"top"}},"id":"9ccfdd0e-2895-4f93-b5bb-b0f46e0d1dba","type":"Quad"},{"attributes":{"callback":null,"column_names":["left","right","top"],"data":{"left":{"__ndarray__":"Cvm3za5PEMCZHufvxesPwB5LXkQuOA/AonfVmJaEDsAnpEzt/tANwKzQw0FnHQ3AMf06ls9pDMC2KbLqN7YLwDpWKT+gAgvAv4KgkwhPCsBErxfocJsJwMnbjjzZ5wjATggGkUE0CMDSNH3lqYAHwFhh9DkSzQbA3I1rjnoZBsBhuuLi4mUFwObmWTdLsgTAahPRi7P+A8DwP0jgG0sDwHRsvzSElwLA+Zg2iezjAcB+xa3dVDABwAPyJDK9fADAED04DUuS/78Ylia2Gyv+vyLvFF/sw/y/LEgDCL1c+782ofGwjfX5v0D631lejvi/SFPOAi8n979SrLyr/7/1v1wFq1TQWPS/Zl6Z/aDx8r9wt4emcYrxv3gQdk9CI/C/BNPI8CV47b8YhaVCx6nqvyw3gpRo2+e/QOle5gkN5b9Qmzs4qz7iv8iaMBSZ4N6/8P7pt9tD2b8YY6NbHqfTv4COuf7BFMy/0FYsRkfbwL+AfHw2Moemv4BiuKu4XqY/QFB74yjRwD8AiAicowrMP+DfSioPotM/sHuRhsw+2T+QF9jiidveP7BZj58jPOI/oKeyTYIK5T+Q9dX74NjnP3hD+ak/p+o/aJEcWJ517T+o7x+D/iHwP6CWMdotifE/mD1DMV3w8j+M5FSIjFf0P4SLZt+7vvU/eDJ4Nusl9z9w2YmNGo34P2iAm+RJ9Pk/XCetO3lb+z9Uzr6SqML8P0h10OnXKf4/QBziQAeR/z+c4flLG3wAQBa1gveyLwFAkogLo0rjAUAMXJRO4pYCQIgvHfp5SgNABAOmpRH+A0B+1i5RqbEEQPqpt/xAZQVAdH1AqNgYBkDwUMlTcMwGQGwkUv8HgAdA5vfaqp8zCEBiy2NWN+cIQNye7AHPmglAWHJ1rWZOCkDSRf5Y/gELQE4ZhwSWtQtAyuwPsC1pDEBEwJhbxRwNQMCTIQdd0A1AOmeqsvSDDkC2OjNejDcPQDAOvAkk6w9A1nCi2l1PEECU2mawKakQQFJEK4b1AhFADq7vW8FcEUDMF7QxjbYRQIqBeAdZEBJASOs83SRqEkA=","dtype":"float64","shape":[100]},"right":{"__ndarray__":"mR7n78XrD8AeS15ELjgPwKJ31ZiWhA7AJ6RM7f7QDcCs0MNBZx0NwDH9OpbPaQzAtimy6je2C8A6Vik/oAILwL+CoJMITwrARK8X6HCbCcDJ24482ecIwE4IBpFBNAjA0jR95amAB8BYYfQ5Es0GwNyNa456GQbAYbri4uJlBcDm5lk3S7IEwGoT0Yuz/gPA8D9I4BtLA8B0bL80hJcCwPmYNons4wHAfsWt3VQwAcAD8iQyvXwAwBA9OA1Lkv+/GJYmthsr/r8i7xRf7MP8vyxIAwi9XPu/NqHxsI31+b9A+t9ZXo74v0hTzgIvJ/e/Uqy8q/+/9b9cBatU0Fj0v2Zemf2g8fK/cLeHpnGK8b94EHZPQiPwvwTTyPAleO2/GIWlQsep6r8sN4KUaNvnv0DpXuYJDeW/UJs7OKs+4r/ImjAUmeDev/D+6bfbQ9m/GGOjWx6n07+Ajrn+wRTMv9BWLEZH28C/gHx8NjKHpr+AYriruF6mP0BQe+Mo0cA/AIgInKMKzD/g30oqD6LTP7B7kYbMPtk/kBfY4onb3j+wWY+fIzziP6Cnsk2CCuU/kPXV++DY5z94Q/mpP6fqP2iRHFiede0/qO8fg/4h8D+gljHaLYnxP5g9QzFd8PI/jORUiIxX9D+Ei2bfu771P3gyeDbrJfc/cNmJjRqN+D9ogJvkSfT5P1wnrTt5W/s/VM6+kqjC/D9IddDp1yn+P0Ac4kAHkf8/nOH5Sxt8AEAWtYL3si8BQJKIC6NK4wFADFyUTuKWAkCILx36eUoDQAQDpqUR/gNAftYuUamxBED6qbf8QGUFQHR9QKjYGAZA8FDJU3DMBkBsJFL/B4AHQOb32qqfMwhAYstjVjfnCEDcnuwBz5oJQFhyda1mTgpA0kX+WP4BC0BOGYcElrULQMrsD7AtaQxARMCYW8UcDUDAkyEHXdANQDpnqrL0gw5AtjozXow3D0AwDrwJJOsPQNZwotpdTxBAlNpmsCmpEEBSRCuG9QIRQA6u71vBXBFAzBe0MY22EUCKgXgHWRASQEjrPN0kahJABlUBs/DDEkA=","dtype":"float64","shape":[100]},"top":{"__ndarray__":"s6auGMn1ZT+zpq4YyfVlPwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALOmrhjJ9WU/AAAAAAAAAAAAAAAAAAAAALOmrhjJ9WU/k6auGMn1ZT8AAAAAAAAAAJOmrhjJ9XU/AAAAAAAAAAAAAAAAAAAAAJOmrhjJ9WU/0qauGMn1dT+Tpq4YyfV1P7OmrhjJ9YU/Bf2C0lZ4gD+zpq4YyfWFP7OmrhjJ9WU/wdGY9Q83kz9fUNpeO3ObPwX9gtJWeJA/s6auGMn1lT/d0Zj1DzeTP8HRmPUPN6M/9GVPzd4Tqj+Je8Q7grSoP/RlT83eE6o/O3JIGwUosT9zD3sTUZGvP/RlT83eE7o/9GVPzd4Tuj+ht+X2LdDAP+gbdGF3pcY/2lQY73k5zz+nXNOsYYfSP6wBwWKVPtQ/xJV3OmQb2z9Z/6EadlviPyTgAZkQXOU/7sBhF6tc6D9YgiEU4F3uP2El8IH0Me4/XSz8TJet6j/wnIMFB5fnP/EzCBg6Kec/9UZePr7m4z89itzRx6vhP+ED4Kq0IdY/9UZePr7m0z+XmrXKouHOP/nrS/TxncU/Zbwjh2yWxD+hOmXwl9K8P/VGXj6+5rM/mzHpzxpGtT/8kDmqJVWnP2W8I4dslqQ/wdGY9Q83oz8qvCOHbJakP/jRmPUPN5M/wdGY9Q83kz8d/YLSVniQP5OmrhjJ9YU/k6auGMn1hT/Spq4YyfVlP5OmrhjJ9WU/0qauGMn1ZT8AAAAAAAAAAJOmrhjJ9WU/0qauGMn1ZT+Tpq4YyfVlP9KmrhjJ9WU/k6auGMn1dT8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADSpq4YyfVlPwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAk6auGMn1ZT8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAk6auGMn1ZT8=","dtype":"float64","shape":[100]}}},"id":"7ffb53d7-2954-42ee-8006-9db7d370a58a","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"d0189600-2bc8-4f4f-8e59-7b111ff4e7ca","type":"LinearAxis"}],"left":[{"id":"873f13d5-ad53-41cc-9e68-558b3625adb1","type":"LinearAxis"}],"renderers":[{"id":"d0189600-2bc8-4f4f-8e59-7b111ff4e7ca","type":"LinearAxis"},{"id":"bd6c6269-c94e-423b-a26b-41c0a1d3da33","type":"Grid"},{"id":"873f13d5-ad53-41cc-9e68-558b3625adb1","type":"LinearAxis"},{"id":"98985a39-f4f7-447e-90da-8df0b5326fe4","type":"Grid"},{"id":"458f8b66-92f2-4945-8953-5ff8544870b6","type":"GlyphRenderer"}],"title":{"id":"5e6aebc1-300c-4468-bae9-518ac28ca463","type":"Title"},"tool_events":{"id":"6fdc9754-e9b6-488e-a2fa-c6b29785e844","type":"ToolEvents"},"toolbar":{"id":"da89543f-9ed9-4bd5-9afc-56343213fc08","type":"Toolbar"},"toolbar_location":"above","x_range":{"id":"339bef66-25cb-4439-919f-916b91ddb00d","type":"DataRange1d"},"y_range":{"id":"176af089-47aa-4cd2-a1f1-bf4fa136c326","type":"DataRange1d"}},"id":"7ae81053-1f24-48a9-85e4-b30704bbc24b","subtype":"Figure","type":"Plot"},{"attributes":{"callback":null},"id":"339bef66-25cb-4439-919f-916b91ddb00d","type":"DataRange1d"},{"attributes":{},"id":"56803c7f-e6cd-4496-8898-175419c75307","type":"BasicTickFormatter"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"29bf3458-2214-4e1c-8371-cf99a3e3d1ee","type":"PanTool"},{"id":"08797199-4344-44d6-a585-ecd6eec0e0e5","type":"WheelZoomTool"},{"id":"c922a12f-3b41-4cfe-ad44-30cb288bd738","type":"ResetTool"},{"id":"c8c8f3b7-92a6-48de-a5a7-a7c8745d236d","type":"SaveTool"}]},"id":"da89543f-9ed9-4bd5-9afc-56343213fc08","type":"Toolbar"},{"attributes":{"formatter":{"id":"771947ef-d589-49ae-98f5-6ef45a8c1024","type":"BasicTickFormatter"},"plot":{"id":"7ae81053-1f24-48a9-85e4-b30704bbc24b","subtype":"Figure","type":"Plot"},"ticker":{"id":"fceb939b-a091-488b-89d4-4a1289254b2b","type":"BasicTicker"}},"id":"d0189600-2bc8-4f4f-8e59-7b111ff4e7ca","type":"LinearAxis"},{"attributes":{},"id":"6fdc9754-e9b6-488e-a2fa-c6b29785e844","type":"ToolEvents"},{"attributes":{"callback":null},"id":"176af089-47aa-4cd2-a1f1-bf4fa136c326","type":"DataRange1d"},{"attributes":{},"id":"fceb939b-a091-488b-89d4-4a1289254b2b","type":"BasicTicker"}],"root_ids":["7ae81053-1f24-48a9-85e4-b30704bbc24b"]},"title":"Bokeh Application","version":"0.12.4"}};
            var render_items = [{"docid":"932bbd7d-9cfa-4f24-a68d-65a2f3b9d077","elementid":"09af843a-8ba0-4f13-a758-59453c5e2213","modelid":"7ae81053-1f24-48a9-85e4-b30704bbc24b"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("09af843a-8ba0-4f13-a758-59453c5e2213")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
frequency_frequency = Counter()

for word, cnt in total_counts.most_common():
    frequency_frequency[cnt] += 1
```


```python
hist, edges = np.histogram(list(map(lambda x:x[1],frequency_frequency.most_common())), density=True, bins=100, normed=True)

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="The frequency distribution of the words in our corpus")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
```




    <div class="bk-root">
        <div class="bk-plotdiv" id="b7f2e30a-9d83-4ebd-a216-d384f5ef713f"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        document.getElementById("b7f2e30a-9d83-4ebd-a216-d384f5ef713f").textContent = "BokehJS successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("b7f2e30a-9d83-4ebd-a216-d384f5ef713f");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid 'b7f2e30a-9d83-4ebd-a216-d384f5ef713f' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"71ebf26b-1b16-419a-bfb4-fb3b12fe4b8e":{"roots":{"references":[{"attributes":{"plot":{"id":"419cc3b9-0bfc-4618-b282-20eddfbff215","subtype":"Figure","type":"Plot"},"ticker":{"id":"80d5047d-6cdd-4f06-a373-f044d2ccf874","type":"BasicTicker"}},"id":"b0566e40-18b7-4f47-9b7f-5b869bd83c36","type":"Grid"},{"attributes":{"formatter":{"id":"e4009fd6-6d49-4acb-96da-1513ad73804e","type":"BasicTickFormatter"},"plot":{"id":"419cc3b9-0bfc-4618-b282-20eddfbff215","subtype":"Figure","type":"Plot"},"ticker":{"id":"841c781a-33a2-4267-b957-c0789a86bd74","type":"BasicTicker"}},"id":"c1a5ddee-bdad-49fd-b476-10a86fd93113","type":"LinearAxis"},{"attributes":{"callback":null},"id":"79df4143-c27c-44df-8806-613cb968be37","type":"DataRange1d"},{"attributes":{"callback":null,"column_names":["left","right","top"],"data":{"left":{"__ndarray__":"AAAAAAAA8D/NzMzMzFhxQM3MzMzMUIFANDMzMzP1iUDNzMzMzEyRQAAAAAAAn5VANDMzMzPxmUBnZmZmZkOeQM3MzMzMSqFAZ2ZmZuZzo0AAAAAAAJ2lQJqZmZkZxqdANDMzMzPvqUDNzMzMTBisQGdmZmZmQa5AAAAAAEA1sEDNzMzMzEmxQJqZmZlZXrJAZ2ZmZuZys0AzMzMzc4e0QAAAAAAAnLVAzczMzIywtkCamZmZGcW3QGdmZmam2bhANDMzMzPuuUAAAAAAwAK7QM3MzMxMF7xAmpmZmdkrvUBnZmZmZkC+QDQzMzPzVL9AAAAAAMA0wEBnZmZmBr/AQM3MzMxMScFAMzMzM5PTwUCamZmZ2V3CQAAAAAAg6MJAZ2ZmZmZyw0DNzMzMrPzDQDMzMzPzhsRAmpmZmTkRxUAAAAAAgJvFQGdmZmbGJcZAzczMzAywxkAzMzMzUzrHQJqZmZmZxMdAAAAAAOBOyEBnZmZmJtnIQM3MzMxsY8lANDMzM7PtyUCamZmZ+XfKQAAAAABAAstAZ2ZmZoaMy0DNzMzMzBbMQDQzMzMTocxAmpmZmVkrzUAAAAAAoLXNQGdmZmbmP85AzczMzCzKzkA0MzMzc1TPQJqZmZm53s9AAAAAAIA00EAzMzMzo3nQQGdmZmbGvtBAmpmZmekD0UDNzMzMDEnRQAAAAAAwjtFAMzMzM1PT0UBnZmZmdhjSQJqZmZmZXdJAzczMzLyi0kAAAAAA4OfSQDMzMzMDLdNAZ2ZmZiZy00CamZmZSbfTQM3MzMxs/NNAAAAAAJBB1EAzMzMzs4bUQGdmZmbWy9RAmpmZmfkQ1UDNzMzMHFbVQAAAAABAm9VAMzMzM2Pg1UBnZmZmhiXWQJqZmZmpatZAzczMzMyv1kAAAAAA8PTWQDMzMzMTOtdAZ2ZmZjZ/10CamZmZWcTXQM3MzMx8CdhAAAAAAKBO2EAzMzMzw5PYQGdmZmbm2NhAmpmZmQke2UDNzMzMLGPZQAAAAABQqNlANDMzM3Pt2UBnZmZmljLaQJqZmZm5d9pAzczMzNy82kA=","dtype":"float64","shape":[100]},"right":{"__ndarray__":"zczMzMxYcUDNzMzMzFCBQDQzMzMz9YlAzczMzMxMkUAAAAAAAJ+VQDQzMzMz8ZlAZ2ZmZmZDnkDNzMzMzEqhQGdmZmbmc6NAAAAAAACdpUCamZmZGcanQDQzMzMz76lAzczMzEwYrEBnZmZmZkGuQAAAAABANbBAzczMzMxJsUCamZmZWV6yQGdmZmbmcrNAMzMzM3OHtEAAAAAAAJy1QM3MzMyMsLZAmpmZmRnFt0BnZmZmptm4QDQzMzMz7rlAAAAAAMACu0DNzMzMTBe8QJqZmZnZK71AZ2ZmZmZAvkA0MzMz81S/QAAAAADANMBAZ2ZmZga/wEDNzMzMTEnBQDMzMzOT08FAmpmZmdldwkAAAAAAIOjCQGdmZmZmcsNAzczMzKz8w0AzMzMz84bEQJqZmZk5EcVAAAAAAICbxUBnZmZmxiXGQM3MzMwMsMZAMzMzM1M6x0CamZmZmcTHQAAAAADgTshAZ2ZmZibZyEDNzMzMbGPJQDQzMzOz7clAmpmZmfl3ykAAAAAAQALLQGdmZmaGjMtAzczMzMwWzEA0MzMzE6HMQJqZmZlZK81AAAAAAKC1zUBnZmZm5j/OQM3MzMwsys5ANDMzM3NUz0CamZmZud7PQAAAAACANNBAMzMzM6N50EBnZmZmxr7QQJqZmZnpA9FAzczMzAxJ0UAAAAAAMI7RQDMzMzNT09FAZ2ZmZnYY0kCamZmZmV3SQM3MzMy8otJAAAAAAODn0kAzMzMzAy3TQGdmZmYmctNAmpmZmUm300DNzMzMbPzTQAAAAACQQdRAMzMzM7OG1EBnZmZm1svUQJqZmZn5ENVAzczMzBxW1UAAAAAAQJvVQDMzMzNj4NVAZ2ZmZoYl1kCamZmZqWrWQM3MzMzMr9ZAAAAAAPD01kAzMzMzEzrXQGdmZmY2f9dAmpmZmVnE10DNzMzMfAnYQAAAAACgTthAMzMzM8OT2EBnZmZm5tjYQJqZmZkJHtlAzczMzCxj2UAAAAAAUKjZQDQzMzNz7dlAZ2ZmZpYy2kCamZmZuXfaQM3MzMzcvNpAAAAAAAAC20A=","dtype":"float64","shape":[100]},"top":{"__ndarray__":"TDhFg5YVbT+JTZDME4n7PtMKDQpDB+Y+1QoNCkMH1j7VCg0KQwfWPgAAAAAAAAAA1QoNCkMHxj7VCg0KQwfGPgAAAAAAAAAA2goNCkMHxj4AAAAAAAAAAAAAAAAAAAAA2goNCkMHxj4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA5QoNCkMHxj4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOUKDQpDB8Y+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA5QoNCkMHxj4=","dtype":"float64","shape":[100]}}},"id":"1dcf734c-3ee8-4820-b255-a6d09b3f6630","type":"ColumnDataSource"},{"attributes":{"plot":null,"text":"The frequency distribution of the words in our corpus"},"id":"32291e3c-3b99-4022-a205-eb50ee2e3791","type":"Title"},{"attributes":{"plot":{"id":"419cc3b9-0bfc-4618-b282-20eddfbff215","subtype":"Figure","type":"Plot"}},"id":"c1650719-b9f0-4219-a6ba-d65320df6105","type":"ResetTool"},{"attributes":{"plot":{"id":"419cc3b9-0bfc-4618-b282-20eddfbff215","subtype":"Figure","type":"Plot"}},"id":"c916bd7e-aa63-4038-a897-ae996b97b15f","type":"SaveTool"},{"attributes":{},"id":"5011db81-ef2c-4076-80e0-1e1ad0235528","type":"ToolEvents"},{"attributes":{"below":[{"id":"f06a62f8-4479-4ddb-8c38-ba4cfd716307","type":"LinearAxis"}],"left":[{"id":"c1a5ddee-bdad-49fd-b476-10a86fd93113","type":"LinearAxis"}],"renderers":[{"id":"f06a62f8-4479-4ddb-8c38-ba4cfd716307","type":"LinearAxis"},{"id":"b0566e40-18b7-4f47-9b7f-5b869bd83c36","type":"Grid"},{"id":"c1a5ddee-bdad-49fd-b476-10a86fd93113","type":"LinearAxis"},{"id":"f46df637-6370-4ad2-8c4c-e660fbdafc69","type":"Grid"},{"id":"0498b1d6-9a31-4c8c-8f03-cf575e7040a5","type":"GlyphRenderer"}],"title":{"id":"32291e3c-3b99-4022-a205-eb50ee2e3791","type":"Title"},"tool_events":{"id":"5011db81-ef2c-4076-80e0-1e1ad0235528","type":"ToolEvents"},"toolbar":{"id":"f0ea0e05-5534-4c03-9a92-502f9ffba541","type":"Toolbar"},"toolbar_location":"above","x_range":{"id":"79df4143-c27c-44df-8806-613cb968be37","type":"DataRange1d"},"y_range":{"id":"51c6ffd2-1fde-4648-89ee-ee00cd39a672","type":"DataRange1d"}},"id":"419cc3b9-0bfc-4618-b282-20eddfbff215","subtype":"Figure","type":"Plot"},{"attributes":{"formatter":{"id":"4dc029e2-a490-4afe-9d77-8d9d7ee08260","type":"BasicTickFormatter"},"plot":{"id":"419cc3b9-0bfc-4618-b282-20eddfbff215","subtype":"Figure","type":"Plot"},"ticker":{"id":"80d5047d-6cdd-4f06-a373-f044d2ccf874","type":"BasicTicker"}},"id":"f06a62f8-4479-4ddb-8c38-ba4cfd716307","type":"LinearAxis"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"8d2223a7-1e09-4dec-bddc-b1ee2cf4c729","type":"PanTool"},{"id":"5bead03e-6766-4265-abc1-96776fa9c176","type":"WheelZoomTool"},{"id":"c1650719-b9f0-4219-a6ba-d65320df6105","type":"ResetTool"},{"id":"c916bd7e-aa63-4038-a897-ae996b97b15f","type":"SaveTool"}]},"id":"f0ea0e05-5534-4c03-9a92-502f9ffba541","type":"Toolbar"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"right":{"field":"right"},"top":{"field":"top"}},"id":"aecb1e69-3dd3-4661-9a1c-d12b0e5174f4","type":"Quad"},{"attributes":{"callback":null},"id":"51c6ffd2-1fde-4648-89ee-ee00cd39a672","type":"DataRange1d"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#1f77b4"},"left":{"field":"left"},"line_color":{"value":"#555555"},"right":{"field":"right"},"top":{"field":"top"}},"id":"73151402-98e3-4d15-b1c1-9905289b200d","type":"Quad"},{"attributes":{},"id":"80d5047d-6cdd-4f06-a373-f044d2ccf874","type":"BasicTicker"},{"attributes":{},"id":"841c781a-33a2-4267-b957-c0789a86bd74","type":"BasicTicker"},{"attributes":{"dimension":1,"plot":{"id":"419cc3b9-0bfc-4618-b282-20eddfbff215","subtype":"Figure","type":"Plot"},"ticker":{"id":"841c781a-33a2-4267-b957-c0789a86bd74","type":"BasicTicker"}},"id":"f46df637-6370-4ad2-8c4c-e660fbdafc69","type":"Grid"},{"attributes":{},"id":"4dc029e2-a490-4afe-9d77-8d9d7ee08260","type":"BasicTickFormatter"},{"attributes":{},"id":"e4009fd6-6d49-4acb-96da-1513ad73804e","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"1dcf734c-3ee8-4820-b255-a6d09b3f6630","type":"ColumnDataSource"},"glyph":{"id":"73151402-98e3-4d15-b1c1-9905289b200d","type":"Quad"},"hover_glyph":null,"nonselection_glyph":{"id":"aecb1e69-3dd3-4661-9a1c-d12b0e5174f4","type":"Quad"},"selection_glyph":null},"id":"0498b1d6-9a31-4c8c-8f03-cf575e7040a5","type":"GlyphRenderer"},{"attributes":{"plot":{"id":"419cc3b9-0bfc-4618-b282-20eddfbff215","subtype":"Figure","type":"Plot"}},"id":"5bead03e-6766-4265-abc1-96776fa9c176","type":"WheelZoomTool"},{"attributes":{"plot":{"id":"419cc3b9-0bfc-4618-b282-20eddfbff215","subtype":"Figure","type":"Plot"}},"id":"8d2223a7-1e09-4dec-bddc-b1ee2cf4c729","type":"PanTool"}],"root_ids":["419cc3b9-0bfc-4618-b282-20eddfbff215"]},"title":"Bokeh Application","version":"0.12.4"}};
            var render_items = [{"docid":"71ebf26b-1b16-419a-bfb4-fb3b12fe4b8e","elementid":"b7f2e30a-9d83-4ebd-a216-d384f5ef713f","modelid":"419cc3b9-0bfc-4618-b282-20eddfbff215"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("b7f2e30a-9d83-4ebd-a216-d384f5ef713f")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python

```


```python
import time
import sys

# Let's tweak our network from before to model these phenomena
class SentimentNetwork:
    def __init__(self, reviews,labels, min_count = 10, polarity_cutoff = 0.1, hidden_nodes = 10, learning_rate = 0.1):
       
        np.random.seed(1)
    
        self.pre_process_data(reviews, polarity_cutoff, min_count)
        
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)
        
        
    def pre_process_data(self,reviews):
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
         
    def pre_process_data(self, revies, polarity_cutoff, min_count):
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()
        
        for i in range(len(reviews)):
            if(labels[i] == 'POSITIVE'):
                for word in reviews[i].split(' '):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(' '):
                    negative_counts[word] += 1
                    total_counts[word] += 1
                    
        pos_neg_ratios = Counter()
        
        for term, cnt in list(total_counts.most_common()):
            if(cnt >= 50):
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
                pos_neg_ratios[term] = pos_neg_ratio
        
        for word, ratio in pos_neg_ratios.most_common():
            if(ratio > 1):
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log(1 / (ratio + 0.01))
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):
                if(total_counts[word] > min_count):
                    if(word in pos_neg_ratios.keys()):
                        if((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                            review_vocab.add(word)
                        else:
                            review_vocab.add(word)
        
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
            
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
            
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
            
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1,input_nodes))
        self.layer_1 = np.zeros((1,hidden_nodes))
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def update_input_layer(self,review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(" "):
            self.layer_0[0][self.word2index[word]] = 1

    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def train(self, training_reviews_raw, training_labels):
        
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))
        
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer

            # Hidden layer
#             layer_1 = self.layer_0.dot(self.weights_0_1)
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
            
            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            if(np.abs(layer_2_error) < 0.5):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
        
    
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                            + " #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        
        # Input Layer


        # Hidden layer
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
        
```


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], min_count=20, polarity_cutoff=0.05, learning_rate=0.01)
```


```python
mlp.train(reviews[:-1000],labels[:-1000])

```

    Progress:99.9% Speed(reviews/sec):2550. #Correct:20282 #Trained:24000 Training Accuracy:84.5%


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):3239. #Correct:855 #Tested:1000 Testing Accuracy:85.5%


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], min_count=20, polarity_cutoff=0.8, learning_rate=0.01)
```


```python
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:99.9% Speed(reviews/sec):2566. #Correct:20282 #Trained:24000 Training Accuracy:84.5%


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):3238. #Correct:855 #Tested:1000 Testing Accuracy:85.5%


---
博客地址：[52ml.me](http://www.52ml.me)
---

