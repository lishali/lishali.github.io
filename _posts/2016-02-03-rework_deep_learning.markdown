---
layout: post
title:  “Deep Learning and Startups“
date:   2016-02-03 15:04:23
categories: [jekyll]
tags: [Deep Learning, Tech]
---


**This blog post is also featured in [KDnuggets][KD]**
[KD]: http://www.kdnuggets.com/2016/02/deep-learning-startups-rework-san-francisco.html


## **General Thoughts**


This past week I went to the [Rework Deep Learning][1] conference.  It was a good two days of talks by both top researchers in DL and companies applying DL.  I made summary notes for the talks and divided them by ‘Research’ and ‘Companies’. Within these two categories they are simply in order of who talked first.  There was also a Q&A with Andrew Ng which I stuffed under 'Research'.  

The research talks were an excellent line up.  All presented on recent work, though if you keep up with the literature, they should be familiar.  I won't mention any in particular here because I think their summaries are all worth reading if you are not yet familiar.

There is clearly a lot of excitement about this area of research for its applications, the companies at this conference is that they can be generally divided into the following categories.  The fairly large ones (Pinterest, Twitter, eBay, Flickr…etc) use DL to be competitive.  It is another tool in the bag of various other statistical/machine learning techniques they were already using, but can work well with the deluge of visual data and natural language processing tasks they must do at scale. I.e they already have a product, DL makes it marginally or perhaps much better.  

Of the smaller companies, some with medical applications had very clear and tractable problems (see for instance BayLabs below for ultrasound diagnosis) along with a well defined market.  These companies mainly take advantage of the advances in computer vision to do classification, which can make diagnosis both more efficient and more accessible.  This seems convincing as long as there is a well labelled set for the problem, and in the case of ultrasound diagnosis, partnering with hospitals and research institutions seems to provide just that. Nonetheless, the regulatory tangle that surrounds businesses in healthcare push such businesses (at least in the near future) to assume an assistance role, so that the responsibility of diagnosis/practice still falls with a healthcare practitioner. 

There were also two companies focusing on the analysis of satellite images.  One can imagine how valuable this data is once obtained (counting world oil supplies is one good example, as well as measuring construction activity in China as a proxy for the Chinese economy).  So there is certainly a market (ranging from hedge funds to even non-profits).  The price paid for the data is another issue, though as companies like SpaceX lower the cost of space tech, we can expect growth in the availability of private suppliers of satellite data (for better or for worse), so a business founded on this supply is a good start.  Finally, there were some traditional consumer facing verticals that try to use DL to improve the personalization aspect of human-computer interaction (e.g in shopping and recommendations, organizing your photo library...etc).  These verticals seem more challenging to me as standalone businesses; An issue to consider in creating a successful business using DL is to find a vertical that is ok with the accuracy level that is achieved by the state of the art, while being willing to pay a lot (profitable amounts) of money for it. In short are there people who care enough given the achieved level of accuracy to pay for your product? Though a lot of the companies look promising, it is not obvious they have succeeded yet.  Moving away from specific verticals, it seems easier to be business facing and build a platform to do DL, as many of the companies below do, especially since the companies that can afford it have already built DL teams (large companies mentioned above).  

All in all I find this area of entrepreneurial activity very exciting and I plan on posting more analysis on startups in this space on my blog.  

Keep in mind the notes are a bit unpolished as I took them in real time for all the talks.  


[1]: https://www.re-work.co/events/deep-learning-sanfran-2016
[2]: https://www.stat.berkeley.edu/~janetlishali/


## **Research**

### **Speaker/Company name: Ilya Sutskever/OpenAI**

Theme of presentation: Deep Learning Theory and Applications

Technology advancements: Deep Neural Networks (DNN) are very expressive and can represent solutions to very difficult problems we are interested in.  They are also trainable from actual data (that is, we can find good weights).  Speaker gives a soft argument by analogy with biological neurons to convince ourselves about why neural nets should be able to solve similar problems biological neural nets solve (biological neurons fire slowly a small number of times in a second).  Presented on new research coined “Neural GPU” to address the issue of how to make DNN generalize better by studying how they learn algorithms.  Results:  can train binary number addition of up to 20 bits and generalize to 2000 bits, same for multiplication.  Also highly parallel.  
 
Key take-away: DNNs strike a sweet spot between trainability and expressiveness. 


### **Speaker/Company name: Fireside Chat with Andrew Ng.**

Main things discussed: we need to work together to educate the community on what is possible and what is not, so that there is no artificial scare.  Andrew is very optimistic about the technology, and also about how the community approaches it in an open sourced way.  There needs to be more of a role from government, but not in constraining solutions.  Rather their role should be focusing on setting safety standards.

More in depth summary (paraphrased): 
Optimistic about self-driving.  Automakers increasing levels of autonomy (but there is a chasm).  Start with a bus route (more feasible to make it safe), then grow from that.  Passionate about role of government in AD (autonomous driving).  Get behind and help grow this system.  

Question: Viability, private and public attacks?

Answer:  Distinguish between come and stop (vision cannot distinguish at this point).  So need improvements.  But cars have no blind spots.  So there are better and worse of self driving cars (manage our expectations, i.e should be different).  Analogy: trains are more dangerous than horses.  Society had to understand things not to do in front of trains.  Similarly to autonomous cars.  Want people to recognize the car and be primed to expect different behaviours (so physically make the car distinctive).  Not the prettiest but most easily recognizable.  See the world different than people (ACars).

Q: Anticipate accident from AI.  If there is a PR backlash, how to respond. 

A:  Be unapologetically optimistic about AI.  Be transparent about what AI can and cannot do.  Fears that reckless company that release features that are not ready.  Careful, honest, open at explain to regulators and public.  

Q: What legislation do you support and not support?

A:  Regulating safety, but putting a human in the car to achieve that is unimaginative.  Get a standard rather than a specific solution (such as putting a human in the car).  

Q: pledge for safe AI.  Trusting companies?

A:  These decision makers are just people in this room.  Optimistic about technology.  

Q: Musk fear of AI, where does this stem from?

A:  AI so smart in so many verticals, so many data we have from these problems, has risen very rapidly.  Gap between AI and what humans do, so we don’t have the appropriately formulated data for supervised learning.  So the value created from AI is from supervised learning.  Tempting to exponentially extrapolate, but amount of data that needs to be formatted is not growing exponentially so fear is very far.  

Q: Projecting, what are the AI ethical issues in the 5 year out in the future? 

A:  AI create so much value.  So much bigger than any company (google, baidu, fb).  Eg.  Iphone transformed our technology.  Less worried about problems.  Growing the ecosystem.  Have everyone benefit from a piece of it.  Encourage the growth.  

Q:  Narrative is an AI race?  You don’t see it as that? 

A:  Sure we compete for talent.  We are at the start of growing an ecosystem, so all need to work to make it bigger.  Shifting from software and algorithms to data (company focus shift).  Open publishing papers.  Role of open source software.  

Q:  How do you convince businesses to participate in the open approach?

A:  Good PR, accrue talent.  Yes we need to make profit, but it is aligned with helping silicon valley can lift together.  Eg.  Coursera.  Wouldn’t be this excited about job if I didn’t really believe in it.  

Q:  Changes in IP regulations to help this vision?  

A:  Patent system is badly broken.  Valuable IP is data.  Value captured in short amount of time, so hence broken patent system.  In addition, for data, what is the social contract?  Complicated conversation.  

Q: Affects of AI on jobs.  How should businesses think about this?  You? Role of government? 

A:  AI will create massive labour displacement (putting people out of jobs).  Truck drivers for instance.  Want to encourage public education of what AI is.  The more we can educate, the higher chance we can make better decisions.  Coursera is a positive impact (so educate people whose jobs we displace).  Basic income (he supports).  Or negative income tax?  More discourse on this.  Supports university income for studying so we have high chance of entering work force.  





### **Speaker/Company name: Christopher Manning/Prof. CS&Linguistics at Stanford**

Theme of presentation: Deep Learning the Precise Meaning of Language.  

Technology advancements: Ratios of co-occurrence probabilities can encode meaning components and give great gains over Categorical CRF, SVD, C&W (other popular methods).  There remains the hard problems of compositionality.  I.e, how do we go from understanding smaller things to larger things?  They have exploited transfer learning adaptation and ensemble modeling (leveraging DL) to gain 25% improvement of translating TED talks.  Now developing representations of universal dependencies (universal across languages) with neural network based dependency parser.  

Key take-away:  Language is the way we transfer knowledge over time and space.  If we start to unlock NL-understanding, we must allow computers to learn from our wealth of digitized resources. 




### **Speaker/Company name: Andrej Karpathy/PhD Student at Stanford**

Theme of presentation: Visualizing and Understanding Recurrent Networks

Recurrent Neural Networks (RNNs) offer flexibility.  Instead of allowing for a fixed input and output size, we input sequences and output sequences.  Great for sentiment classification, videos (frames--so no arbitrary cutoff for what the output must be a function of).  

Great rhetorical flourish: "“Generate, sample or hallucinate the output.”

Example applications:  Generate poetry via feeding in character level RNN of the works of Shakespeare. generate Latex documents of Algebraic Geometry research almost complies., generate  C code.  All really good at capturing the statistical patterns of data.  

Requires only 112 lines of Python code (char-rnn (Torch7))!

Currently studying how this works, and how things change at scale.  There are interpretable cells in the RNN.  Eg.  Ones that are for quote detection, detect new lines.  Also implemented, combine RNN with CNN.  Gives a good description of images (also funny failure cases).  Another use is to query image database with text. ConNet, and RNN, stuck those blocks together on images. Test image, then feed it through CNN, instead of a classifier at the end, we redirect the rep through the RNN, give prob of first word in the description.  Using data from Amazon Mechanical Turk.  Failure cases.  Extended model tackling joint task of both detection and description.  Fully convolutional, fully differentiable.  Query with small piece of text, and then look through images for that query. 

Check out his class at Stanford with Fei Fei Li and Justin Johnson at http://cs231n.stanford.edu/ .



### **Speaker/Company name: Ian Goodfellow/Research Scientist Google**

Theme of presentation: Tutorial on Optimization for Deep Networks

Best practices and how they work:  The correct metaphor when thinking about analyzing the contribution of each layer of a NN is to think of NN as a sequence of matrix multiplications (one matrix multiplication per level).  Of course, it is difficult with an increase in layers to figure out the effect of one layer.  It turns out that in practice, the error surfaces has a thin area of minimal error, and very large slope around it, which makes it easy to overshoot.  

Objections:  There are no mathematical guarantees to how we find good minima (of course not a convex surface!).  However, plotting error over time in the application cases gets us a surprisingly smooth and monotonic curve.  

Theoretical developments:  Batch Normalization--normalized by mean and sd so that your gradient descent can easily adjust to differences in how each layer affects output.  In practice, this makes training more easy and faster.  




### **Speaker/Company name: Roland Memisevic/Assistant Professor University of Montreal**

Theme of presentation: Deep Learning as a Compute Paradigm. 

Switch from CPUs to GPUs allowed us to leverage the level of parallelization allowed in NN and made current success possible.  How can we go a step further and make it more efficient on the level of hardware?  DL paradigm.  Specific, more accurate but more data needed.  Generic, less accurate but less data needed.  Classic paradigm:  specific, faster but tedious to program, generic: slower but easy to program.  In sum, DL is more about accuracy than specificity.  
Humans are way more on the generic side.  Use very little data, and solve things in generality.  Think embodied cognition.  Humans are champions of solving new verticals with premade architecture for another task.  

Key take-away: We should model the hardware to exploit the parallel operations in neural networks.  
Also check out Twenty Billion Neurons (new startup).  





### **Panel on How Important will Deep Learning Applications be for Future Economic Growth.**

Concern for malevolent AI, hype or reality?  Answer: Hype, a nice news story.  

DL divides the Haves and have notes, will it’s progress worsen this divide?  Answer:  DL is really open sourced in nature, so actually way more accessible for companies than before.  The rising tide is for specialists to be able to focus on their speciality than having to reinvent the wheel everytime.  This enables them to do more.  Concluding thoughts:  echo Andrew Ng’s opinion that we need for regulation of safety rather than solutions.  Government is not the expert, so should make guidelines not prescribe what a solution should be.  




### **Speaker/Company name: Oriol Vinyals/Google DeepMind**

Theme of presentation: Sequence to Sequence Learning @Google Brain.

Technology advancements: Language modeling using RNN.  Thanks to LSTM, Theano, Torch, implementation is easier than ever.  Sequence to sequence learning allows us to try many new things, and get better at image captions, speech, parsing, dialogue, video generation, algorithm (learning of).  Key to success is to believe it will work and work hard at optimizing parameters.  

Key take-away: Sequences have become first class citizens.  With enough data, they can make very complicated functions.  Attention made recurrent models even better.





### **Speaker/Company name: Aditya Khosla/PhD Student MIT**

Theme of presentation: Understanding human behaviour through visual media.  

Can we predict what kind of images people will like or remember?   Can we predict personality from a picture?  Someone’s political inclinations through an image?  Can we predict a person’s state of mind?  

Technology:  Developed iTracker CNN to figure out where people are looking.  Within 2cm of error, expect it to get even better.  Demo can be found at: http://gazefollow.csail.mit.edu  

Application:  Good for diagnosis for autism and schizophrenia.  Obviously also advertising.  





### **Pieter Abbeel**
Associate professor at Berkeley EECS

Deep reinforcement learning for robots.  

What is the standard approach for robots.

Percepts (accelerometer), hand engineered state estimate, hand generated control policy class, hand-tuned 10is free parameters, motor commands.  Replace 3 middle steps with DNN.  This mirrors revolution in vision and speech recognition.  

But not the same, because robotics is NOT a supervised learning problem.  Robotics has feedback loop.  Sparse reward function (cook dinner, after present it we give stars).  

3 challenges: stability.  Veer off track as a car, then we are far from what we from data is not represented.   Sparse supervision.  

Deep Reinforcement learning locomotion.  Give it reward, how hard we hit the ground, how far forward we get.  transferable via different types of robots.  Very flexible.  


Frontiers/future.  Memory, estimation, temporal hierarchy/goal setting.  shared and transfer learning.  




### **Lise Getoor, University of California, Santa Cruz**
Scalable Collection Reasoning in Graphs.  

Setting: structured data.  “Big Data is not flat”  Multimodal.  spatio-temporal.  
Take nicely structured data, flatten into table, and do not leverage the structure.  

NEED: ML for Graphs.  Input graphs.  (not just sequences).  

Challenges:  components are inter and intra dependent.  

Goal: input graph infer an output graph.  

Key idea:  predictions/outputs depend on each other joint reasoning is required.  Challenge,  Really large and really loopy graphs.  

Tool: Probabilistic Soft Logic.  A declarative probabilistic programming language for collective inference problems on richly structured graph data.  Solves most of these problems.  

Summarize:  Need techniques to deal with structured data.  New opportunities for algos that deal with this.  Their tool to tackle this is PSL.  getoor@usc.edu 






## **Companies**

### **Speaker/Company name: Diogo Moitinho de Almeida/Enlitic**

Theme of presentation: The tools for Deep Learning (DL) are hard to manipulate, we need to work together to make it more flexible.   For instance, adding layers to networks is easy because the tools make it easy, but what about other things?  Hard things:  costs with parameters, nodes with non-differentiable updates.  And in general, we need to develop general “tricks” beyond the ad hoc bag of tricks to implement DL.  

Technology advancements: Working on a library to make things more composable, and dynamic architectures possible.  It has been very useful for several medical diagnosis tasks.  

Key take-away: DL is awesome, tools limit their development, so let’s work together to fix that!





### **Speaker/Company name: Clement Farabet/ Twitter**

Theme of presentation: Deep Learning at Twitter

Applications for business: Timeline ranking/personalization, content discovery and suggestion, Periscope ranking/personalization, search, indexing and retrieval, Ads targeting and click through rate prediction.  Platform safety.  All these are made harder by needing to function in real time.  

Challenges:  Nothing is generic, transfer learning is difficult.  Need a solid loop of data collection, fast at scale.  Also using something trained from another data set is not that applicable (ImageNet); Twitter for instance, has very few natural images.  

Technology advancement: Developed three packages to address issues of fast model exploration, how to handle distributed structured datasets, and distributed training.  They are troch-autograd, torch-datasets and torch-distlearn respectively.  





### **Speaker/Company name: Matthew Zeiler/Clarifai**

Theme of presentation: Forevery: Deep Learning for Everyone.  Want to understand images and videos.  Thier technology can power many different verticals: Vimeo, Pixar, medical devices from iphones taking pictures inside of ear, nose and mouths..etc.  So both enterprise and consumer.  

Example of technology in use.  New Forevery app.  Organizes photos by people, place/time and things.  No manual tag content needed.  Predicts new tags in real time, established concept categories (eg. adventure, summer, relaxation).  Learn as I click to give it feedback on the right pictures.  

Key take-away:  Deep learning allows us to achieve a new level of personalization.  







### **Speaker/Company name: Charles Cadieu/Postdoctoral Research at MI/CEO Bay Labs**

Theme of presentation: Increasing Quality, Value and Access to Medical Imaging

Applications for business:  Bring to more people and situations an increased value and quality for medical image readings.  For instance, in ultrasound imaging.  Great application area because it can be widely used (no ionizing radiation), it has comparable or superior diagnostic capability to other tools, it’s affordable (comparatively) and can be quite portable with current technologies (via tablet and iphone plugins).  Been able to apply DL to detect Rheumatic heart disease.  

Key take-away:  Turn DL to understand the world in us, and use hardware advancements to make it accessible.  





### **Speaker/Company name: Andrew Zhai/Pinterest**

Theme of presentation: Visual Search at Pinterest

Technology advancements: Flashlight: want Pinterest to identify the objects in an image and recommend similar objects.  Technical challenges: image retrieval over billions of images in less than 250ms.  DL used in image representation and retrievals.  Take CNN’s intermediate layers and use their embeddings as a measure of similarity.  Scaling up is another challenge.  Solution:  divide set of images into shards, then do KNN in a distributed infrastructure.  Really leveraged up on the open source libraries available (Caffe).  

Room for improvement:  Putting a default crop that starts on an objects in the picture than an hardcoded predetermined position.  





### **Speaker/Company name: Pierre Garrigues/Flickr**

Theme of presentation:Deep Learning with Flickr Tags

Applications for business: User tag prediction.  Photo search and discovery.  Photo aesthetic models (to judge how good a photo is for recommendation).  The state of the art is ImageNet, however, photos on Flickr have non literal labels (such as one describing the image of an ocean in terms of the way it makes one feel).  This gives a very rich and different data set.  

Challenges: Model exploration and needing to train models fast.  Some solutions:  Spark on yarn.  Caffe on Spark.






### **Speaker/Company name: Neil Glackin/Intelligent Voice**

Theme of presentation: Recurrent Lateral Spiking Networks for Speech Enhancement

Technology advancements: Noisy environments (outside) where voice recognition fares less well.  

Biological inspiration: Our cochlea is a fast fourier transform that digitizes sound to spiking stimulus.  
Spiking Neuron:  Laterally Recurrent, symmetrical connectivity parametrized by two parameters.  Really good at cleaning noise from sound.  






### **Speaker/Company name: Matthias Dantone, Fashwell**

Theme of presentation:A Deep Learning Pipeline to Convert Fashion Images into Valuable Fashion Assets

Image processing using DL.  Make instagram shoppable.  Product leverages the dataset on instagram to make recommendations on things that people can not even articulate they want.  Need to identify the fashion objects in an image, and must recommend the correct item to users, also providing direct link to their product in partner web-shop.







### **Speaker/Company name:Steven Brumby, Co-Founder & CTO, Descartes Labs**

Theme of presentation:Seeing the Earth from the Cloud: How Machine Learning is Changing the Way we See the World.

Take advantage of efficiency of current DL code to analyze the massive amounts of data we have on the earth via satellite images of agricultural data.  Application example:  commodities traders would be interested in getting a much more refined idea of what sort of food is grown, where, how much, and the import export activity.  With the help of DL to analyze this massive dataset, Descartes Labs can provide this type of insight.  






### **Speaker/Company name:Eli David, CTO, Deep Instinct @deepinstinctsec**

Deep Learning for security: analyzes apk (mobile apps) that finds/identifies malicious ones. Shows very strong results compared to some traditional methods.

Key take away: Deep Instinct brings a new approach to cybersecurity by using DL to identify cyber-attacks and block them in real-time before any real harm can occur.

Speaker/Company name: Maithili Founder CEO of Sightline Innovation.
Deep Learning for detecting defects on manufactured parts (e.g. car parts). Some special optical cameras take images of parts as they are manufactured and then Deep Learning does quality control.
Can take the same technology and apply to medical diagnostics. 
Largest Machine Learning startup in Canada.








### **Richard Socher**
CTO and CEO, and Co-founder Metamind
From Classification to Question Answering for Language and Vision

If done right, standard image and text classification can be very accurate.  
Application: Intracranial Hemorrhage: classifying thousands of CT scans to prioritize what are bad cases of hemorrhage.  

The future:  NLP/AI tasks can be reduced to question answering.  But not all problems are as simple as classification.  I.e Machine translation, sentiment analysis 

Goal:  Joint model for QA answering.  Problem: for NLP no single model architecture with consistent state of the art results across tasks.  Problem II:  joint multitask learning hard.  

Solutions: Dynamic Memory Networks developed at Metamind.  Uses gated recurrent units in RNN.  Includes reset gate that determines whether we should care about what we know in the past.  Can ignore irrelevant information. Episodic Memory module.  Gates are activated if relevant to the question (terms are similarity functions).  Inspiration from Neuroscience: Episodic memory is the memory of autobiographical events (times, places..etc).  The seat of episodic memory (hippocampus) is active during transitive inference.  

Result: Most accurate model for visual question answering in the world.  Examples:  What sport is shown: tennis.  What is the food, is it healthy, calories...etc so not strict classification.  






### **Kaheer Suleman CTO and CO-Founder**
Maluuba

Dialogue. Natural language understanding.  Join ConvNet and LSTM.  Beyond Domains: Machine Reading Comprehension.  Types forms of answer in multiple choice, and sees if machine chooses the right one.  Achieved 78% (best) for machine comprehension algorithms.  The data set of human short stories and multiple answers.   

Working on methods to learn more general tasks. Using DL net, variance reduced. 

Future of conversational systems:
domain independence (health, games, twitter...etc)
 common sense reasoning.  No good dataset to learn common sense.  
 sense of humor, multimodal.  






### **Arno Candel**
Cheif Architect H20.ai

Product:  H20 Software, optimized Java code.  Runs anywhere (laptop/server/cluster).  Hadoop and Spark, open source.  Client APIs, R, Python, Java, Scala, Flow.  

Application: Deep Learning aces particle physics.  Detect Higgs Bosons from background Signal.  Beats all other models.   

Key take away:  Compatible with many different languages, really efficient Java code, great way to bring DL to many verticals.  

Nigel Duffy CTO of AI, Sentient Technologies 

DL will transform online shopping experience.  For instance buying shoes is hard.  Use DL to better this recommendation/match process.  Visual discovery, we don’t know exactly what we want.  Dialog based, images only, fun, productive, and works on any device.  20 questions to get what you want to buy.  

How:  Embedding using CNN to capture semantics of the products.  Use amazon mechanical turk for data set labelling.  Their own on shoes.com

Take away:  Make online shopping fun via dialog.  Deep learning makes this possible, .  






### **Hassan Sawaf/Ebay**


800 million listing
266 million app downloads
Use cases of DL and NN in ebay:
-Machine translation (semantic feature extraction)
-NLP spelling and synonyms
- Information extraction
	-product classification
	-attribute extraction
-visual object recognition
-spam identification 
-visual search






### **Davide Morelli**

CTO Biobeats

Measuring heart rate.  
Gather cardiovascular data in the world (from mobile phone).  Make music in synch with the heart.  Build ML infra to make sense of heart rate data.  User value:  get dashboard.  Shows insights and data.  confluence of data from all apps.  Stress, fatigue, resilience, sleep data.  Numbers, versus widgets.  

Building several apps.  GetOnUP.  dynamic radio, music depending on walking/running.  Sync the rhythm with your rhythm (soundtrack of your day). Make it fun so they get data.  

Hear and Now: Meditation app.  Breathing exercises.  MEasures HRV response.  Teach users to breathe.  Intervention engine: should you take a break now.  immediate biofeedback  Train neural network to analyze data

Human activity recognition module.  

Conceptual challenges.  Variable way that our body reacts.  Context is paramount, Hard to get a ground truth.  Latent variables, physiological, physiological, personalized baseline is needed.  





### **Alex Jaimes CTO at AICure**

We all had to take medication, and not as prescribed 
Founded 5 years ago, tested, funding from NIH.  Series A.  Now scaling and growing revenue.  

Where adherence matters.  Clinical trials.  Population health (high risk, high cost conditions) Health care 18% of GDP.  Clinical trials %51 billion a year.  

Technology: scalable, interactive, inexpensive, any language, adaptive and personalized, sensitive to language and computer literacy levels.  Real time interventions...big data insights.  

continuous data points versus sporadic blood tests. Nonadherence is costly. 

Goal: intelligent medical assistant.  






### **James Crawford, Founder Orbital Insight**

How to take millions of satellite images and understand the earth.  Commercial space.  Growth in huge and small satellites.   50X increase in satellite imagery by 2020. We don’t want pixels, we want insights.  Their product is to convert satellite images to insights.  

Application examples: 
-U.S. Retail:  Patterns for when to go to places (shopping concentration).  peak on wednesdays for Kohl's.  Got the peaks in time (time series of what is going on).  Classifying pixels.  based on carness of each pixel.  

-World Oil Inventory:  Floating lid (it falls and rises due to how much in tank….actually safer.  But great for image analysis.  Proprietary algorithms to find the shadows and find the inventory.  Stabilize oil prices.  These tanks are owned by many different companies, so we don’t know the totality of how much oil.  

Poverty Mapping:  Count cars, better than survey data.  Paid project with the World Bank.  Applicable in multiple verticals.  

Summary:  Orbital insight is for decision makers who want advanced, unbiased knowledge of socioeconomic trends.  Do this via computer vision and cloud computing to turn millions of images into a big-picture understanding of the world that is quantitatively grounded in observation.  






### **Autonomous Action Taking and the need for HAI**
Luca Rigazio

Autonomous action taking.  With robots, we seem to have less agency.  Huge opportunities, but also risks that are difficult to qualify and quantify.  Two successes in history.  HTF.  Airliners → autopilots.  

Example:  pilots crashed the airplane (france) by fighting autopilot.  

Human AI = HAI 

How to get the final layer, feeling.  
Questions left for audience:  Only understanding or synthesis?  Is it needed to operate well with humans, have or display.  What and how to measure.  

Florentine.  Agents fully learned from data.  


<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-73314404-1', 'auto');
  ga('send', 'pageview');

</script>



