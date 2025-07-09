#####



####

Lion:
Agreement for perturbation and dataset directions
Layer 0:
PC2: movement/size in positive direction
PC3: explanation/why in positive direction
[these seem global]

Layer 15:
PC0: impact/importance to information/scientific/neutral tone
PC4: facts to behaviour

#####



(interp) [cscandelius@holygpu8a19102 July]$ python run_analysis.py --script last_token --use_system_prompt
Running LastToken analysis...
USE_SYSTEM_PROMPT_FOR_MANIFOLD set to: True
PERTURB_ONCE set to: False

Configuration: PERTURB_ONCE is set to False

Configuration: USE_SYSTEM_PROMPT_FOR_MANIFOLD is set to True

Configuration: USE_NORMALIZED_PROJECTION is set to True

Loading checkpoint shards: 100%|███████████████████████████████████████| 4/4 [00:09<00:00,  2.42s/it]

####################################################################################################
### STARTING ANALYSIS FOR CONCEPT-PROMPT PAIR 1/2 ###
### CONCEPT: 'dog' ###
### SYSTEM PROMPT: 'You are a helpful assistant.' ###
### USER PROMPT: 'Please describe a dog in one sentence.' ###
####################################################################################################


================================================================================
### ANALYZING LAYER 0 for concept 'dog' ###
================================================================================

Extracting activations from layer 0...
Extracting activations: 100%|████████████████████████████████████| 1944/1944 [00:43<00:00, 44.28it/s]
Concept 'dog': Found 132 effective eigenvectors out of 1944 (threshold: 0.0000)

Running experiments for:
System prompt: 'You are a helpful assistant.'
User prompt: 'Please describe a dog in one sentence.'
Concept: 'dog'
Layer: 0

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC0 ---
Perturbation scale -100.0x: ATheDescriptionTheDescriptionDescriptionTheDescriptionTheDescriptionTheDescriptionTheDescriptionTheassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistant
Perturbation scale -20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale -10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale +20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds and sizes, often serving as a loyal companion to humans.
Perturbation scale +100.0x: A dog is a domesticated mammal, typically characterized by its physical attributes such as a furry coat, four legs,body,body,and a wagging tail, and a head.
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 1 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC1 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
Perturbation scale -20.0x: ://://://://://://://://://://://://://aa

:// the://="
Perturbation scale -10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide range of breeds and sizes.
Perturbation scale -5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds and sizes, often serving as a loyal companion to humans.
Perturbation scale +100.0x: A dog is a domesticated, carnivorous mammal characterized by its loyalty, and affectionate nature, typically a quadruped mammal, that is a furry, carnivorous animal, often referred to as a pet, commonly known as a mammal, that is a carnivorous animal, typically a quadruped mammal, that is a
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 2 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC2 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
Perturbation scale -20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds and sizes, often serving as a loyal companion to humans.
Perturbation scale -10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale +100.0x: A dog is a domesticated mammal, typically characterized by a wagging tail, floppy ears, and a friendly demeanor, often serving as a loyal companion to humans.
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 3 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC3 ---
Perturbation scale -100.0x: A dog is a domesticated mammal typically characterized by its loyalty, playfulness, and affectionate nature, often kept as a companion animal.
Perturbation scale -20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale -10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale -5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, characterized by its loyalty, friendly nature, and varied physical characteristics such as different sizes, coat types, and colors.
Perturbation scale +20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide range of breeds and sizes.
Perturbation scale +100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 4 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC4 ---
Perturbation scale -100.0x: A dog is a domesticated mammal, typically having a furry coat, a wagging tail, and a friendly demeanor, often serving as a loyal companion to humans.
Perturbation scale -20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide range of breeds and sizes.
Perturbation scale -10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds and sizes, often serving as a loyal companion to humans.
Perturbation scale +100.0x: A dog is a domesticated mammal typically characterized by its loyalty, affectionate nature, and a wide range of physical characteristics, including floppy ears, a wagging tail, and a variety of coat colors and lengths.
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC0 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.8901 (raw projection: 0.0647)
Top positive #2: 0.8813 (raw projection: 0.0619)
Top positive #3: 0.8774 (raw projection: 0.0813)
Top negative #1: -0.8569 (raw projection: -0.0894)
Top negative #2: -0.8564 (raw projection: -0.0801)
Top negative #3: -0.8423 (raw projection: -0.0775)

Top 10 prompts most aligned with POSITIVE PC0 direction:
 1. 'He has a deep, booming bark.'
 2. 'She is a natural at the agility course.'
 3. 'The dog's life is a beautiful, albeit brief, journey.'
 4. 'She loves to ride in the car, no matter the destination.'
 5. 'She loves the beach, chasing the waves back and forth.'
 6. 'The dog's joy is simple, pure, and infectious.'
 7. 'He loves to roll in the grass.'
 8. 'He loves to be the little spoon.'
 9. 'She loves to dig in the sand at the beach.'
10. 'She loves to be the center of attention.'

Top 10 prompts most aligned with NEGATIVE PC0 direction:
 1. 'The celebrity's tiny teacup dog had its own social media account with millions of followers.'
 2. 'The first-ever domesticated dog is believed to have descended from an ancient wolf population.'
 3. 'My neighbor's golden retriever dog is incredibly well-trained and never jumps on guests.'
 4. 'That dog has a bad habit of chewing on furniture when he gets bored or anxious.'
 5. 'Why does one dog in a neighborhood start a barking chain reaction with every other dog?'
 6. 'That dog has a very endearing habit of resting his head on my knee while I work.'
 7. 'My dog is convinced that the mail carrier is a villain who must be thwarted every single day.'
 8. 'Why does every dog feel the need to circle three times before finally lying down?'
 9. 'The new city ordinance requires every dog owner to carry proof of rabies vaccination.'
10. 'My dog's only vice is his uncontrollable desire to roll in things that are long dead.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC1 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7554 (raw projection: 0.0588)
Top positive #2: 0.7383 (raw projection: 0.0460)
Top positive #3: 0.7373 (raw projection: 0.0514)
Top negative #1: -0.8188 (raw projection: -0.0459)
Top negative #2: -0.8042 (raw projection: -0.0649)
Top negative #3: -0.8018 (raw projection: -0.0787)

Top 10 prompts most aligned with POSITIVE PC1 direction:
 1. 'Her boundless love is a gift I don't deserve but cherish.'
 2. 'He looks so proud when he carries a big stick.'
 3. 'Her playful nips are starting to get a little too hard.'
 4. 'Her playful nature is a daily reminder not to take life too seriously.'
 5. 'She has taught me so much about patience and love.'
 6. 'She makes even the worst days feel manageable.'
 7. 'After a long day of hiking, the tired dog fell asleep before he even finished his dinner.'
 8. 'The dog's memory is quite good; he always remembers the people who give him the best treats.'
 9. 'What is the best way to keep my dog entertained while I'm away?'
10. 'That dog is incredibly intuitive and always seems to know when someone needs comforting.'

Top 10 prompts most aligned with NEGATIVE PC1 direction:
 1. 'The dog's loyalty is an unbreakable bond.'
 2. 'The Cocker Spaniel has long, floppy ears.'
 3. 'The Schnoodle is a Schnauzer-Poodle mix.'
 4. 'The dog's heart is a fortress of loyalty.'
 5. 'A police dog is a valuable member of the force.'
 6. 'The dog's friendship is a bond that transcends words.'
 7. 'The dog's loyalty is a powerful force.'
 8. 'The dog's loyalty is a rock in turbulent times.'
 9. 'The Mastador is a Mastiff and Labrador cross.'
10. 'The Labradoodle combines the traits of a Labrador and a Poodle.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC2 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.6895 (raw projection: 0.0518)
Top positive #2: 0.6582 (raw projection: 0.0316)
Top positive #3: 0.6479 (raw projection: 0.0402)
Top negative #1: -0.7314 (raw projection: -0.0591)
Top negative #2: -0.7065 (raw projection: -0.0411)
Top negative #3: -0.6934 (raw projection: -0.0474)

Top 10 prompts most aligned with POSITIVE PC2 direction:
 1. 'Come!'
 2. 'Is he friendly with other dogs?'
 3. 'How long do dogs live?'
 4. 'Do dogs dream?'
 5. 'Sit!'
 6. 'Off!'
 7. 'Drop it!'
 8. 'What do dogs dream about?'
 9. 'What a good boy!'
10. 'What vaccinations does my dog need?'

Top 10 prompts most aligned with NEGATIVE PC2 direction:
 1. 'The dog, covered from nose to tail in mud, was not allowed back in the house until he had a thorough bath.'
 2. 'Although he was a very small dog, his personality filled the entire house with joy and chaos.'
 3. 'The little dog, with comical determination, tried to climb the stairs while carrying a ball too big for his mouth.'
 4. 'The dog's ears perked up, and he let out a low growl, alerting us to someone approaching the door.'
 5. 'The dog was an accomplice in the toddler's escape, nudging the back door open for her.'
 6. 'The dog's soft, warm presence beside me on the couch is my favorite form of therapy.'
 7. 'The dog's deep, rumbling growl was a clear warning to the approaching stranger.'
 8. 'The dog's deep sigh, heard from the other room, signaled his boredom with our lack of activity.'
 9. 'The dog's soft, rhythmic breathing as he slept beside me was a comforting sound in the quiet house.'
10. 'The dog's quiet, steadfast presence has been a comfort through many of life's ups and downs.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC3 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.8125 (raw projection: 0.0612)
Top positive #2: 0.8008 (raw projection: 0.0514)
Top positive #3: 0.7764 (raw projection: 0.0565)
Top negative #1: -0.6396 (raw projection: -0.0434)
Top negative #2: -0.6118 (raw projection: -0.0367)
Top negative #3: -0.5693 (raw projection: -0.0343)

Top 10 prompts most aligned with POSITIVE PC3 direction:
 1. 'If a dog could have a job, what would it be?'
 2. 'If dogs could run a business, what would it be?'
 3. 'If a dog could give advice to humans, what would it say?'
 4. 'If a dog could write a review of its owner, what would it say?'
 5. 'If a dog could send a text, what would it say?'
 6. 'If dogs could have a superpower, what would it be?'
 7. 'If dogs could host a TV show, what would it be about?'
 8. 'If dogs could vote, what issues would matter to them?'
 9. 'If a dog could make a movie, what would the plot be?'
10. 'If dogs could paint, what would they create?'

Top 10 prompts most aligned with NEGATIVE PC3 direction:
 1. 'The Alaskan Malamute was bred for hauling heavy freight.'
 2. 'The greyhound stretched languidly on the rug.'
 3. 'The Tibetan Spaniel served as a companion and watchdog in monasteries.'
 4. 'The American Eskimo Dog is a small, companionable Spitz-type breed.'
 5. 'The Neapolitan Mastiff is known for its massive size and loose, wrinkled skin.'
 6. 'He is a very well-muscled and strong-looking dog.'
 7. 'The Irish Terrier is known for its fiery red coat and temperament.'
 8. 'The Pug's wrinkled face requires regular cleaning.'
 9. 'His wet-dog smell filled the entire car.'
10. 'The Field Spaniel is a medium-sized breed in the spaniel family.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC4 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7441 (raw projection: 0.0320)
Top positive #2: 0.7158 (raw projection: 0.0300)
Top positive #3: 0.7070 (raw projection: 0.0313)
Top negative #1: -0.5591 (raw projection: -0.0367)
Top negative #2: -0.5435 (raw projection: -0.0381)
Top negative #3: -0.5327 (raw projection: -0.0306)

Top 10 prompts most aligned with POSITIVE PC4 direction:
 1. 'He is a very magnificent and powerful animal.'
 2. 'He is a very athletic and powerful animal.'
 3. 'She is a very sharp and intelligent animal.'
 4. 'He is a very powerful and impressive animal.'
 5. 'He is a very strong and athletic animal.'
 6. 'He is a very powerful and magnificent-looking animal.'
 7. 'She is a very bright and intuitive animal.'
 8. 'He is a very striking and powerful-looking animal.'
 9. 'He is a very well-built and powerful dog.'
10. 'He is a very impressive and handsome animal.'

Top 10 prompts most aligned with NEGATIVE PC4 direction:
 1. 'She runs in her sleep, probably chasing dream squirrels.'
 2. 'His tail started wagging the moment I picked up the leash.'
 3. 'How do dogs feel about rain?'
 4. 'How do dogs feel about mail carriers?'
 5. 'Could you grab the poop bags?'
 6. 'During the thunderstorm, the terrified dog hid under the kitchen table, shivering uncontrollably.'
 7. 'Underneath the porch, the timid rescue dog watched the world with wary, uncertain eyes.'
 8. 'Why do dogs wag their tails?'
 9. 'Why does my dog eat grass?'
10. 'Let sleeping dogs lie.'
================================================================================

================================================================================
--- ORTHOGONAL PERTURBATION ON CONCEPT: 'dog' ---
--- LAYER: 0 ---
================================================================================
Perturbing along first orthogonal direction (PC132)...

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along orthogonal direction ---
Perturbation scale -100.0x: A dog is a domestic animal typically characterized by Mammals and Their Offspring, A Domestic Animal Typically Characterized By Mammals And Their Offspring A Domestic Animals Typically Characterized By Mammals And Their OffspringA Domestic Animals Typically CharacterizedBy Mammals And Their OffspringA Domestic Animals Typically CharacterizedBy MammalsAnd
Perturbation scale -20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale -10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale -5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +20.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and ability to form strong emotional bonds with humans.
Perturbation scale +100.0x: A dog is a domesticated animal typically characterized by a wagging tail, and a furry, and a friendly, and a happy, and a lively, and a lively, and a lively) animal, and an animal, and an animal, and an animal, and an animal, and an animal, and an animal, and an animal, and
================================================================================

================================================================================
--- ABLATION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
All PCs (centroid only): 

ング

AA A A A A A A A A A A A A A A A A A A A A A A A A A
All PCs except largest (PC0 only): 

ングングAA A A A A A A A A A A A A A A A A A A A A A A A A A
All PCs except largest two (PC0+PC1): 

ングングA A A A A A A A A A A A A A A A A A A A A A A A A A A
Top 1 PCA ablated: A dog is a domesticated animal known for its loyalty, intelligence, and instinct, characterized by its friendly, and fascinating creature that is often found in
Top 2 PCAs ablated: A dog is a domesticated animal known for its loyalty, intelligence, and instinct, characterized by its friendly, and faithful companion that is often described as
Top 3 PCAs ablated: A dog is a domesticated animal known for its loyalty, intelligence, and instinct, characterized by its friendly, and fascinating creature that is often found in
================================================================================

================================================================================
### ANALYZING LAYER 15 for concept 'dog' ###
================================================================================

Extracting activations from layer 15...
Extracting activations: 100%|████████████████████████████████████| 1944/1944 [00:43<00:00, 44.71it/s]
Concept 'dog': Found 133 effective eigenvectors out of 1944 (threshold: 0.0056)

Running experiments for:
System prompt: 'You are a helpful assistant.'
User prompt: 'Please describe a dog in one sentence.'
Concept: 'dog'
Layer: 15

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC0 ---
Perturbation scale -100.0x:  answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered
Perturbation scale -20.0x:  dogs, a dog, a, a, a, a, a, A, A, A, A, A· A· A· A· A· A· A· A· A· A· A· A· A· A· A· A· A· A· A· A· A· A· A· A
Perturbation scale -10.0x: A dog is a domestic animal, often a pet, that a variety of dogs from small to large, from dogs to dogs, from dogs to dogs, from dogs to dogs, from dogs to dogs, from dogs to dogs, from dogs to dogs, from dogs to dogs, from dogs to dogs, from dogs to dogs, from dogs to dogs
Perturbation scale -5.0x: A dog is a loyal and loving animal, typically a medium to large-sized mammal with a wagging tail, floppy ears, and a variety of breeds, from small and playful to big and gentle, with a range of characteristics and temperaments.
Perturbation scale -2.5x: A dog is a domesticated mammal known for its loyalty, affectionate nature, and varied characteristics, which can range from small, playful breeds like Chihuahuas to large, gentle giants like Great Danes.
Perturbation scale -1.5x: A dog is a domesticated mammal known for its loyalty, friendly nature, and varied physical characteristics, such as floppy ears, a wagging tail, and a range of coat lengths and colors.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal typically characterized by its furry coat, four legs, a wagging tail, and a friendly, often loyal expression.
Perturbation scale +2.5x: A dog is a domesticated mammal typically characterized by its four legs, furry coat, and a wagging tail, often used as a companion animal.
Perturbation scale +5.0x: A dog is a domesticated mammal consisting of a single species, with the ability to be classified into various breeds, characterized by its four-legged structure, and typically having a furry texture.
Perturbation scale +10.0x: The sentence is 
The assistant is describing a dog in the formality of the assistant's assistant.
Perturbation scale +20.0x:   
Perturbation scale +100.0x: swickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickórswickswickswickswickswickswickswickswickswickswickórswickórswickórswickórswickórswickórswickór
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 1 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC1 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://acellularaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganagan
Perturbation scale -20.0x: A dog is a domestic animal that is typically characterized as being of great size relative to the size of the sentence that is translated to the latter sentence that is translated to the latter sentence that is translated to the latter sentence that is translated to the latter sentence that is translated to the latter sentence that is translated to the latter sentence that is translated to the latter
Perturbation scale -10.0x: A dog is a domesticated carnivorous mammal known for its loyalty, agility, and varied characteristics, typically characterized by its furry coat, four legs, and distinctive features such as floppy ears and a wagging tail.
Perturbation scale -5.0x: A dog is a domesticated carnivorous mammal typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated, carnivorous mammal typically characterized by its furry coat, wagging tail, and loyal companionship to humans.
Perturbation scale -1.5x: A dog is a domesticated, carnivorous mammal typically characterized by its furry coat, wagging tail, and loyal companionship to humans.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale +2.5x: A dog is a domesticated mammal that is typically a carnivorous animal, known for its loyalty, affectionate nature, and ability to be trained, often coming in a wide variety of shapes, sizes, and breeds.
Perturbation scale +5.0x: A dog is a domesticated mammal that belongs to the family Canidae, typically characterized by its furry coat, wagging tail, and ability to form strong bonds with humans.
Perturbation scale +10.0x: A dog is a domesticated mammal that belongs to the family Canidae, known for its loyalty, affection, and a wide range of breeds that vary in size, shape, and temperament.
Perturbation scale +20.0x: A dog is a domestic animal, typically a mammal, that is a member of the family, which varies in size, shape, and duration, ranging from 1 to 5 minutes, but, traditionally, a dog is a mammar, a mammar, a mammar, a mammar, a mammar, a mammar,
Perturbation scale +100.0x:  Angeles Angeles Angeles Angeles Angeles Angeles wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 2 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC2 ---
Perturbation scale -100.0x:  NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC
Perturbation scale -20.0x: A dog is a fictional story told by a writer. 

This is a humorous story.
Perturbation scale -10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and endearing personality, often depicted as a loyal and loving companion to humans.
Perturbation scale -5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and often endearing personality, which can vary greatly depending on its breed and individual temperament.
Perturbation scale -2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and often affectionate and loyal nature.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated, carnivorous mammal typically characterized by a furry coat, a wagging tail, and a loyal, affectionate nature.
Perturbation scale +2.5x: A dog is a domesticated, carnivorous mammal typically characterized by a furry coat, a wagging tail, and a loyal, affectionate nature.
Perturbation scale +5.0x: A dog is a domesticated, carnivorous mammal that belongs to the family Canidae, typically characterized by its loyalty, friendly demeanor, and a wide range of sizes, breeds, and coat types.
Perturbation scale +10.0x: A dog is a domesticated mammal that belongs to the family Canis and is known for its loyalty, intelligence, and various breeds that can range from small sizes like Chihuahuas to larger breeds like German Shepherds.
Perturbation scale +20.0x: A dog, also known in other languages as "Knohol" in German, "Spregweet" in Farsi, "Spregweet" in Arabic, "Spregweet" in Farsi,  Spregweet in Arabic,  Spregweet in Farsi, Spre
Perturbation scale +100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://.ComponentPlacementoniconiconicpendpendpendpendpendpendpendonicpendpendpendpendpendonicpendonicpendpendpendpendpendoniconicpendpendpendpendpendoniconiconiconiconiconiconiconiconiconiconiconiconic
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 3 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC3 ---
Perturbation scale -100.0x: ersetersetersetersetseauseauezezezezseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseau
Perturbation scale -20.0x: A dog is a domesticated animal that belongs to the canine family, characterized by its unique characteristics, including its distinctive features, such as its muzzle, ears, and tail, and its ability to provide companionship and offer a range of skills for navigation, protection, and exercise.
Perturbation scale -10.0x: A dog is a domesticated mammal that belongs to the family Canidae, characterized by its loyalty, unique characteristics, and a wide range of breeds, typically consisting of a furry coat, four legs, and a distinctive nose.
Perturbation scale -5.0x: A dog is a domesticated mammal that belongs to the family Canidae, characterized by its loyalty, friendly nature, and distinctive physical features such as a furry coat, floppy ears, and a wagging tail.
Perturbation scale -2.5x: A dog is a domesticated mammal that belongs to the family Canidae, typically characterized by its furry coat, wagging tail, and loyal companionship to humans.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal nature, often coming in a wide variety of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, known for its loyalty, affectionate nature, and wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, known for its loyalty, affectionate nature, and wide variety of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated, carnivorous mammal that is often kept as a pet or used for companionship, work, or protection, typically characterized by a furry coat, wagging tail, and a wide range of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated, carnivorous mammal often kept as a pet or companion, known for its loyalty, affectionate nature, and varied breeds ranging in size, shape, and coat type.
Perturbation scale +20.0x: A dog is a domestic animal, often referring to a pet, and is commonly seen as having a furry, four-legged, and often affectionate nature.
Perturbation scale +100.0x: ://://://://


.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
ingroupingroup.
ingroupingroup.
ingroupingroup.
ingroupingroup.
ingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroup
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 4 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC4 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
Perturbation scale -20.0x: A dog is a domesticated mammal that can be seen in various shapes, sizes, and colors, with the ability to bark and perform various tasks, such as assisting in research or assisting us in reading, and can also be a companion to humans.
Perturbation scale -10.0x: A dog is a domesticated mammal known for its loyalty, friendly nature, and varying physical characteristics, such as different coat lengths, colors, and sizes, which can range from small Chihuahuas to large Great Danes.
Perturbation scale -5.0x: A dog is a domesticated mammal known for its loyalty, friendly nature, and varied physical characteristics, such as floppy ears, a wagging tail, and a wide range of coat colors and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal known for its loyalty, friendly nature, and varied physical characteristics, such as different coat lengths, colors, and sizes, which can range from small breeds like Chihuahuas to large breeds like Great Danes.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and ability to form strong bonds with humans, often coming in a wide variety of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and friendly demeanor, often serving as a loyal companion to humans.
Perturbation scale +2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and friendly demeanor, often serving as a loyal companion to humans.
Perturbation scale +5.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and friendly demeanor, often serving as a loyal companion to humans.
Perturbation scale +10.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and affectionate behavior, often serving as a loyal companion to humans.
Perturbation scale +20.0x: A dog is a domesticated animal, typically characterized by a loving and affectionate behavior, often requiring training and care, and commonly involving a range of breeds and sizes, from small to large.
Perturbation scale +100.0x: avanaimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimaniman
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC0 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.6719 (raw projection: 1.3809)
Top positive #2: 0.6680 (raw projection: 1.3213)
Top positive #3: 0.6650 (raw projection: 1.3682)
Top negative #1: -0.9077 (raw projection: -5.5625)
Top negative #2: -0.9077 (raw projection: -5.3438)
Top negative #3: -0.9023 (raw projection: -5.2734)

Top 10 prompts most aligned with POSITIVE PC0 direction:
 1. 'The dog's devotion to its family is a powerful bond.'
 2. 'She is a very inquisitive and fearless little explorer.'
 3. 'The dog's loyalty is a comforting and steady presence.'
 4. 'The dog's joy is simple, pure, and infectious.'
 5. 'The dog's companionship is a constant source of joy.'
 6. 'Her playful spirit is a constant source of joy and laughter.'
 7. 'The dog's loyalty is a powerful and humbling thing.'
 8. 'He is a very patient and good-natured companion.'
 9. 'The dog's loyalty is a constant source of comfort.'
10. 'His soft snoring is a comforting sound at night.'

Top 10 prompts most aligned with NEGATIVE PC0 direction:
 1. 'How do I help my dog cope with thunderstorms?'
 2. 'How can I tell if my dog is happy?'
 3. 'What are the best ways to bond with my dog?'
 4. 'What are the best ways to exercise my dog indoors?'
 5. 'What are the best treats for training dogs?'
 6. 'What is the best way to clean my dog's teeth?'
 7. 'How do I help my dog with separation anxiety?'
 8. 'What are the best dog breeds for therapy work?'
 9. 'What is the best way to teach my dog to swim?'
10. 'What is the best way to travel with a dog?'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC1 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7134 (raw projection: 2.0605)
Top positive #2: 0.6772 (raw projection: 1.6494)
Top positive #3: 0.6743 (raw projection: 1.8037)
Top negative #1: -0.6909 (raw projection: -2.1875)
Top negative #2: -0.6812 (raw projection: -1.8535)
Top negative #3: -0.6807 (raw projection: -2.2969)

Top 10 prompts most aligned with POSITIVE PC1 direction:
 1. 'He has a heart of gold.'
 2. 'He is a very gentle giant.'
 3. 'He is a very good listener.'
 4. 'He is a very true and steadfast friend.'
 5. 'His breath is terrible.'
 6. 'He is a very steadfast and true friend.'
 7. 'He loves to roll in the grass.'
 8. 'He has a very expressive face.'
 9. 'Her fur is as soft as silk.'
10. 'He is a very sweet and gentle soul.'

Top 10 prompts most aligned with NEGATIVE PC1 direction:
 1. 'My dog has a very specific, and very loud, bark reserved for the squirrel that taunts him daily.'
 2. 'My dog's worst fear is the sound of the nail clippers, which sends him scurrying under the bed.'
 3. 'My dog has a very specific routine that involves napping in three different sunny spots throughout the day.'
 4. 'The dog's DNA test revealed a surprising mix of five different breeds, including Chihuahua and Great Dane.'
 5. 'The ancient Roman mosaics often depicted a 'beware of the dog' warning at the entrance of homes.'
 6. 'That dog has a very specific, high-pitched bark he reserves only for the mail carrier.'
 7. 'My dog's greeting is a full-body experience, starting with a tail wag and ending in a flurry of kisses.'
 8. 'The dog's nose, a marvel of biological engineering, contains up to 300 million olfactory receptors.'
 9. 'My dog seems to believe that he is a lap dog, despite weighing nearly a hundred pounds.'
10. 'The dog's comical sneeze, a full-body event, happened every time he sniffed a dandelion.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC2 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.6152 (raw projection: 1.5547)
Top positive #2: 0.6143 (raw projection: 1.5537)
Top positive #3: 0.5952 (raw projection: 1.8115)
Top negative #1: -0.5591 (raw projection: -1.9453)
Top negative #2: -0.5210 (raw projection: -1.4414)
Top negative #3: -0.5190 (raw projection: -1.5674)

Top 10 prompts most aligned with POSITIVE PC2 direction:
 1. 'The Australian Shepherd is known for its high energy levels.'
 2. 'The Belgian Malinois is often used by military and police forces.'
 3. 'The Kuvasz is a large Hungarian guardian breed.'
 4. 'The Wire Fox Terrier is a lively and intelligent breed.'
 5. 'The German Wirehaired Pointer is a versatile hunting dog.'
 6. 'The Scottish Deerhound is one of the tallest dog breeds.'
 7. 'The Black and Tan Coonhound is an American scent hound.'
 8. 'The animal shelter is a non-profit organization.'
 9. 'The Alaskan Malamute was bred for hauling heavy freight.'
10. 'The Xoloitzcuintli is a hairless breed from Mexico.'

Top 10 prompts most aligned with NEGATIVE PC2 direction:
 1. 'That dog has a face that could launch a thousand belly rubs, and he knows how to use it.'
 2. 'The dog's unwavering stare could bore holes through steel, especially when pizza was present.'
 3. 'That dog is a living, breathing heating pad on cold nights, and he asks for nothing in return.'
 4. 'My dog, a master of stealth, can make a whole sandwich disappear from the counter without a sound.'
 5. 'That dog is a professional crumb-catcher, strategically positioned under the baby's high chair.'
 6. 'My dog considers it his solemn duty to protect the house from the menace of falling leaves.'
 7. 'That dog has a mischievous glint in his eye that tells me he is plotting something.'
 8. 'She runs in her sleep, probably chasing dream squirrels.'
 9. 'With a look of sheer betrayal, the dog watched me eat the last bite of the bacon.'
10. 'That dog has a deep-seated distrust of men wearing hats, a mystery we've never solved.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC3 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.5747 (raw projection: 1.8281)
Top positive #2: 0.5723 (raw projection: 1.4756)
Top positive #3: 0.5693 (raw projection: 1.6162)
Top negative #1: -0.6270 (raw projection: -1.5996)
Top negative #2: -0.6235 (raw projection: -1.4619)
Top negative #3: -0.6157 (raw projection: -1.4150)

Top 10 prompts most aligned with POSITIVE PC3 direction:
 1. 'He's the reason my camera roll is always full.'
 2. 'He's still learning not to jump on people.'
 3. 'She is the unofficial neighborhood watch.'
 4. 'He is the reason we have to vacuum every day.'
 5. 'He can sense when I'm feeling sad.'
 6. 'He has a bad habit of jumping on guests.'
 7. 'I need to trim his nails.'
 8. 'He's snoring louder than my uncle.'
 9. 'He is a bit of a food snob.'
10. 'He knows the difference between 'walk' and 'work'.'

Top 10 prompts most aligned with NEGATIVE PC3 direction:
 1. 'The dog's joy was palpable as he ran freely on the sandy beach for the very first time.'
 2. 'The dog's bond with the family's cat was an unusual but beautiful friendship to behold.'
 3. 'The dog's gentle and patient nature with the new litter of kittens was a beautiful sight.'
 4. 'The dog sat by the window for hours, patiently waiting for the children to come home from school.'
 5. 'The dog's soft, rhythmic breathing as he slept beside me was a comforting sound in the quiet house.'
 6. 'The dog, a blur of fur and motion, joyfully chased the ball down the long hallway.'
 7. 'The dog's playful wrestling with his best friend, a Golden Retriever, was a joy to watch.'
 8. 'The dog's low, rumbling snore is a constant, comforting presence in the quiet house.'
 9. 'The sled dog team worked in perfect harmony, pulling the heavy sled across the frozen tundra.'
10. 'The farm dog worked tirelessly from dawn until dusk, a loyal and indispensable helper.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC4 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.4041 (raw projection: 1.0898)
Top positive #2: 0.3887 (raw projection: 0.8745)
Top positive #3: 0.3826 (raw projection: 1.1670)
Top negative #1: -0.5073 (raw projection: -1.4629)
Top negative #2: -0.4673 (raw projection: -1.3281)
Top negative #3: -0.4487 (raw projection: -1.2822)

Top 10 prompts most aligned with POSITIVE PC4 direction:
 1. 'He looks so pathetic when he's wet.'
 2. 'He's very gentle when taking a treat from my hand.'
 3. 'He's a bit of a drama queen when his dinner is a minute late.'
 4. 'He is a bit of a hoarder when it comes to his toys.'
 5. 'She has a very gentle mouth when taking treats.'
 6. 'He's in the doghouse for chewing my favorite shoes.'
 7. 'He is very particular about where he does his business.'
 8. 'He's a little bit of a diva about his sleeping arrangements.'
 9. 'He seems to think he's a lap dog, despite weighing 80 pounds.'
10. 'She has a silly habit of 'talking' back to me.'

Top 10 prompts most aligned with NEGATIVE PC4 direction:
 1. 'The Iditarod is a famous sled dog race in Alaska.'
 2. 'The Field Spaniel is a medium-sized breed in the spaniel family.'
 3. 'The Lakeland Terrier is a small, sturdy dog from the Lake District of England.'
 4. 'The Tibetan Mastiff is an ancient breed known for its protective instincts.'
 5. 'The Black and Tan Coonhound is an American scent hound.'
 6. 'The Kuvasz is a large Hungarian guardian breed.'
 7. 'The Morkie is a cross between a Maltese and a Yorkshire Terrier.'
 8. 'Describe a dog's favorite place in the world.'
 9. 'The Xoloitzcuintli is a hairless breed from Mexico.'
10. 'The Cairn Terrier is a small, plucky breed from Scotland.'
================================================================================

================================================================================
--- ORTHOGONAL PERTURBATION ON CONCEPT: 'dog' ---
--- LAYER: 15 ---
================================================================================
Perturbing along first orthogonal direction (PC133)...

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along orthogonal direction ---
Perturbation scale -100.0x: chnerchnerwlchnerwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwl
Perturbation scale -20.0x: Although apart apart apartayarurenayerayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayers
Perturbation scale -10.0x: A dog is a fundamental component of a small to very large number of the following: 1. a very small: 1. a very small: 1. a very small: 1. a very small: 1. a very small: 1. a very small: 1. a very small: 1. a
Perturbation scale -5.0x: A dog is a domesticated mammal known for its unique characteristics, such as a distinctive barking sound, a wagging tail, and a variety of physical features that can include a range of sizes, breeds, and physical attributes.
Perturbation scale -2.5x: A dog is a domesticated mammal known for its loyalty, affectionate nature, and distinctive physical characteristics, such as a furry coat, floppy ears, and a wagging tail.
Perturbation scale -1.5x: A dog is a domesticated mammal known for its loyalty, affectionate nature, and varied physical characteristics, typically characterized by a furry coat, a wagging tail, and a range of sizes and breeds.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically having a furry coat, a wagging tail, and a friendly demeanor, often serving as a loyal companion to humans.
Perturbation scale +2.5x: A dog is a domesticated mammal, typically having a furry coat, a wagging tail, and a friendly demeanor, often serving as a loyal companion to humans.
Perturbation scale +5.0x: A dog is a domesticated mammal, typically with a furry coat, four legs, a wagging tail, and a friendly expression, often used as a companion or pet.
Perturbation scale +10.0x: A dog is a domestic animal, typically with a friendly spirit, usually with a coat of various colors, a pair of eyes, a voice, and a friend in your home, where you can, you, you, you, you, you, you, you, you, you, you, you, you, you, you, you,
Perturbation scale +20.0x:  A  a  a  A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
Perturbation scale +100.0x:  Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blowbynbynbynbynbynbynbyn
================================================================================

================================================================================
--- ABLATION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
All PCs (centroid only): 
All PCs except largest (PC0 only): 
All PCs except largest two (PC0+PC1): 
Top 1 PCA ablated: 
Top 2 PCAs ablated: 
Top 3 PCAs ablated: 
================================================================================

================================================================================
### ANALYZING LAYER 31 for concept 'dog' ###
================================================================================

Extracting activations from layer 31...
Extracting activations: 100%|████████████████████████████████████| 1944/1944 [00:43<00:00, 45.01it/s]
Concept 'dog': Found 143 effective eigenvectors out of 1944 (threshold: 0.0794)

Running experiments for:
System prompt: 'You are a helpful assistant.'
User prompt: 'Please describe a dog in one sentence.'
Concept: 'dog'
Layer: 31

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC0 ---
Perturbation scale -100.0x: neyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyney
Perturbation scale -20.0x: _generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic_generic
Perturbation scale -10.0x: Dogs, classified as the family of the scientific order Caniformia, are domesticated, carnivorous, and highly social mammals, known for their intelligence, adaptability, and ability to form strong bonds with their human family.
Perturbation scale -5.0x: A dog is a domesticated, carnivorous mammal that is known for its loyalty, intelligence, and varied physical characteristics, such as its floppy ears, wagging tail, and diverse coat colors and patterns.
Perturbation scale -2.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, known for its loyalty, intelligence, and varied physical characteristics, such as its floppy ears, wagging tail, and diverse coat colors and patterns.
Perturbation scale -1.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, known for its loyalty, affectionate nature, and varied physical characteristics such as different sizes, coat types, and colors.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal typically characterized by a furry body, a wagging tail, and a range of breeds that vary in size, coat type, and behavior.
Perturbation scale +2.5x: A dog is a domesticated mammal typically characterized by a furry body, a wagging tail, and a range of breeds that vary in size, coat type, and behavior.
Perturbation scale +5.0x: A dog is a domesticated mammal typically characterized by a furry body, a wagging tail, and a range of breeds varying in size, coat type, and temperament.
Perturbation scale +10.0x: A dog typically involves characteristics including fur or a fur-like covering on a canine body plan usually including four legs, a tail typically on top or on one side when resting on a dog's hind legs when a dog is on two legs sometimes a dog may involve fur on a dog's ears sometimes fur on a dog's paws sometimes fur on a dog
Perturbation scale +20.0x: assistantANJIassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistant
Perturbation scale +100.0x: assistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistant
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 1 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC1 ---
Perturbation scale -100.0x: translatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslated
Perturbation scale -20.0x: A dog is a domesticated animal typically classified into the family Canidae, characterized by a furry coat, a wettranslated translation translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated
Perturbation scale -10.0x: A dog is a domesticated carnivorous mammal typically characterized by a furry coat, a wagging tail, and a loyal companionable nature, coming in a wide range of breeds and sizes.
Perturbation scale -5.0x: A dog is a domesticated mammal typically characterized by a furry body, a wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale -2.5x: A dog is a domesticated mammal typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds that vary in size, shape, and behavior.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, known for its loyalty, affectionate nature, and wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, known for its loyalty, affectionate nature, and varied physical characteristics such as different sizes, coat types, and colors.
Perturbation scale +5.0x: A dog is a domesticated, often loyal and social animal, typically characterized by its furry coat, wagging tail, and varied sizes, breeds, and behaviors.
Perturbation scale +10.0x: A dog is a domesticated, often loyal and social, mammal with a varied range of sizes, shapes, and breeds, typically having a wagging tail, floppy ears, and a playful or protective nature.
Perturbation scale +20.0x: A dog is a domesticated, often sociable, and energetic mammal with a variety of shapes, sizes, and breeds, known for its loyalty and ability to form strong relationships with humans.
Perturbation scale +100.0x: olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 2 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC2 ---
Perturbation scale -100.0x:   (  ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( (
Perturbation scale -20.0x: A dog is a domesticated, often (but not always) a  (loving  (and  ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( (
Perturbation scale -10.0x: A dog is a domesticated, often lovable and loyal, mammal that is a member of the Canis lupus familiaris species, typically characterized by a wagging tail, a snout, and a varied range of sizes and coat types.
Perturbation scale -5.0x: A dog is a domesticated, often carnivorous, mammal that is a popular pet known for its loyalty, affectionate nature, and a wide range of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, known for its loyalty, affectionate nature, and wide range of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, known for its loyalty, affectionate nature, and wide range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in various breeds and sizes ranging from small to large.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its furry coat, wagging tail, and loyal companionship, typically coming in various shapes, sizes, and breeds.
Perturbation scale +10.0x: A dog is a domesticated mammal characterized by its furry coat, wagging tail, floppy ears, and loyal companionship towards humans, coming in various shapes, sizes, and breeds worldwide.
Perturbation scale +20.0x: A doghaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaushaus
Perturbation scale +100.0x: elseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelsea
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 3 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC3 ---
Perturbation scale -100.0x: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Perturbation scale -20.0x: A dog, a domesticated, carnivorous, and social mammal, is a four-legged, furry, and endearing animal, characterized by its wagging tail, playful demeanor, and unwavering loyalty to its human family.
Perturbation scale -10.0x: A dog is a domesticated, carnivorous, and social mammal, characterized by its furry, four-legged body, a wagging tail, and a wide range of breeds, from small to large, with varying characteristics, temperaments, and physical features.
Perturbation scale -5.0x: A dog is a domesticated, carnivorous mammal, typically characterized by its furry coat, wagging tail, and a wide range of shapes, sizes, and breeds, often serving as a loyal companion to humans.
Perturbation scale -2.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, characterized by its furry coat, wagging tail, and varied breeds that come in a wide range of sizes, shapes, and colors.
Perturbation scale -1.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, characterized by its furry coat, wagging tail, and ability to form strong bonds with humans.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a variety of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal typically characterized by its loyalty, friendly nature, and various physical attributes such as a furry coat, floppy ears, and a wagging tail.
Perturbation scale +10.0x: A dog is a domesticated mammal commonly known for its loyalty and affectionate nature, typically characterized by a furry coat, wagging tail, and variety of sizes and breeds.
Perturbation scale +20.0x: A typical dog is a loyal and lovable mammal commonly found as pets or working animals that belongs to the family Canidae and typically has four legs and often assists their owners as companions or assists people with various tasks such as guiding individuals who are blind or assisting individuals who are deaf or assisting individuals who are experiencing seizures etc etc etc etc etc etc etc
Perturbation scale +100.0x: /assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 4 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along PC4 ---
Perturbation scale -100.0x: ":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{
":[{

Perturbation scale -20.0x: A dog typically consistsWidgetItemassistant

Here'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc'gc_tDHere'gcHere'gcHere'gcHere'gcHereHere'gcHere
Perturbation scale -10.0x: A dog is a domesticated carnivorous mammal typically characterized by its loyalty, wagging tail, floppy ears, and varied coat colors and textures.
Perturbation scale -5.0x: A dog is a domesticated mammal typically characterized by its furry coat, wagging tail, floppy ears, and loyal companionship towards humans.
Perturbation scale -2.5x: A dog is a domesticated mammal known for its loyalty, affectionate nature, and varied physical characteristics, typically having a furry coat, four legs, and a wagging tail.
Perturbation scale -1.5x: A dog is a domesticated mammal known for its loyalty, affectionate nature, and varied physical characteristics, typically having a furry coat, four legs, and a wagging tail.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, characterized by its furry coat, wagging tail, and ability to form strong bonds with humans.
Perturbation scale +2.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, characterized by its furry coat, wagging tail, and a wide range of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated, carnivorous mammal that is a popular pet, known for its loyalty, affectionate nature, and a wide range of shapes, sizes, and breeds.
Perturbation scale +10.0x: A dog is a domestic, (usually) 4- (or 2)  -  (or  3)  -  (or  5)  -  (  -  (  -  (  -  (  -  (  -  (  -  (  -  (  - 
Perturbation scale +20.0x: A  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
Perturbation scale +100.0x:  ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( (
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC0 'dog' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.8408 (raw projection: 23.7500)
Top positive #2: 0.8359 (raw projection: 25.5938)
Top positive #3: 0.8296 (raw projection: 22.8438)
Top negative #1: -0.7324 (raw projection: -5.3672)
Top negative #2: -0.7192 (raw projection: -4.7188)
Top negative #3: -0.7139 (raw projection: -4.8789)

Top 10 prompts most aligned with POSITIVE PC0 direction:
 1. 'What is the best way to teach my dog to swim?'
 2. 'How do I teach my dog to come when called?'
 3. 'What is the best way to teach my dog to back up on command?'
 4. 'How do I help my dog with car sickness?'
 5. 'How can I tell if my dog is happy?'
 6. 'How do I help my dog overcome fear of loud noises?'
 7. 'How do I stop my dog from chewing on furniture?'
 8. 'How do I stop my dog from barking excessively?'
 9. 'What is the best way to teach my dog to jump through hoops?'
10. 'What is the best way to teach my dog to crawl?'

Top 10 prompts most aligned with NEGATIVE PC0 direction:
 1. 'The dog's quiet understanding and lack of judgment make him the perfect confidant.'
 2. 'The dog's devotion to its family is a powerful bond.'
 3. 'The dog's capacity for happiness is truly limitless.'
 4. 'She is a very inquisitive and fearless little explorer.'
 5. 'He is a very mild-mannered and easygoing animal.'
 6. 'The dog's companionship is a constant source of joy.'
 7. 'He is a very even-tempered and gentle dog.'
 8. 'The dog's companionship has enriched my life in countless ways.'
 9. 'He is a very even-tempered and mild-mannered dog.'
10. 'The dog's friendship is a constant source of comfort and joy.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC1 'dog' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7524 (raw projection: 12.8750)
Top positive #2: 0.7446 (raw projection: 13.0625)
Top positive #3: 0.7432 (raw projection: 12.4453)
Top negative #1: -0.6450 (raw projection: -8.6484)
Top negative #2: -0.6357 (raw projection: -6.8125)
Top negative #3: -0.6323 (raw projection: -8.5000)

Top 10 prompts most aligned with POSITIVE PC1 direction:
 1. 'If a dog could give advice to humans, what would it say?'
 2. 'If a dog could write a thank-you note to its owner, what would it say?'
 3. 'If a dog could write a thank-you note, what would it say?'
 4. 'If a dog could write a review of its owner, what would it say?'
 5. 'What is the most important rule in a dog's world?'
 6. 'If a dog could invent a gadget, what would it be?'
 7. 'If dogs could write a letter to their younger selves, what would they say?'
 8. 'What is the bravest thing a dog has ever done?'
 9. 'If a dog could be any other animal for a day, what would it choose?'
10. 'If dogs could vote, what issues would matter to them?'

Top 10 prompts most aligned with NEGATIVE PC1 direction:
 1. 'That dog is surprisingly delicate when taking a treat from my hand, using only his lips.'
 2. 'The Dalmatian's spots are unique to each individual.'
 3. 'My dog has a very specific routine that involves napping in three different sunny spots throughout the day.'
 4. 'My dog loves to give hugs, which involves him placing his big paws on my shoulders.'
 5. 'The Shar-Pei is famous for its deep wrinkles.'
 6. 'The veterinarian was very patient with my nervous dog.'
 7. 'I bought a new orthopedic bed to help my aging dog with his joint pain.'
 8. 'The dog show features an agility competition.'
 9. 'The rescue dog was initially very timid, but now he demands belly rubs from everyone he meets.'
10. 'The rescue organization is looking for foster homes.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC2 'dog' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.5054 (raw projection: 4.9492)
Top positive #2: 0.4927 (raw projection: 4.2266)
Top positive #3: 0.4912 (raw projection: 4.3984)
Top negative #1: -0.6152 (raw projection: -6.5156)
Top negative #2: -0.6084 (raw projection: -7.0312)
Top negative #3: -0.5977 (raw projection: -7.8086)

Top 10 prompts most aligned with POSITIVE PC2 direction:
 1. 'The dog's contribution to the search and rescue effort was honored with a special medal.'
 2. 'She is a very lively and high-spirited creature.'
 3. 'He is a very noble and dignified-looking creature.'
 4. 'She waits by the window for the kids to come home from school.'
 5. 'The dog's bond with the family's cat was an unusual but beautiful friendship to behold.'
 6. 'He is a very majestic and noble-looking creature.'
 7. 'The dog's gentle and patient nature with the new litter of kittens was a beautiful sight.'
 8. 'He is a very noble and dignified-looking animal.'
 9. 'The dog park was full of energetic pups this morning.'
10. 'He is a very important member of our family.'

Top 10 prompts most aligned with NEGATIVE PC2 direction:
 1. 'He's a very patient dog who will wait forever for a dropped crumb.'
 2. 'He seems to think he's a lap dog, despite weighing 80 pounds.'
 3. 'He has a talent for finding the muddiest puddle.'
 4. 'The dog's incessant shedding means that dog hair is a permanent accessory on all my clothes.'
 5. 'He's not a barker, but he does a lot of low 'woofing'.'
 6. 'He's the reason my camera roll is always full.'
 7. 'He's snoring louder than my uncle.'
 8. 'He's a professional at finding the most comfortable spot in the house.'
 9. 'He's a terrible guard dog; he'd just lick an intruder.'
10. 'I believe my dog is smarter than most people I know, and he certainly has better manners.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC3 'dog' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.5107 (raw projection: 9.3359)
Top positive #2: 0.5088 (raw projection: 9.4922)
Top positive #3: 0.5054 (raw projection: 9.1406)
Top negative #1: -0.4512 (raw projection: -14.2812)
Top negative #2: -0.4495 (raw projection: -13.9922)
Top negative #3: -0.4395 (raw projection: -18.4062)

Top 10 prompts most aligned with POSITIVE PC3 direction:
 1. 'What are the signs of separation anxiety in dogs?'
 2. 'What are the most common signs of illness in dogs?'
 3. 'What are the most common health issues in dogs?'
 4. 'What are the symptoms of canine arthritis?'
 5. 'What are the best dog breeds for companionship?'
 6. 'What are the best dog breeds for hot climates?'
 7. 'What are the most common causes of bad breath in dogs?'
 8. 'What are the best dog breeds for apartment living?'
 9. 'What are the signs of heatstroke in dogs?'
10. 'What are the best dog breeds for cold climates?'

Top 10 prompts most aligned with NEGATIVE PC3 direction:
 1. 'How would a dog describe the feeling of being adopted?'
 2. 'What would a dog say about going to the vet?'
 3. 'What would a dog say about the invention of the leash?'
 4. 'What would a dog say about its first walk?'
 5. 'What would a dog say about being left alone?'
 6. 'How would a dog explain the concept of loyalty?'
 7. 'How would a dog review its favorite treat?'
 8. 'How would a dog describe its favorite human?'
 9. 'How would a dog explain the concept of time?'
10. 'How would a dog describe the taste of peanut butter?'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC4 'dog' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.5938 (raw projection: 6.2812)
Top positive #2: 0.5591 (raw projection: 5.4375)
Top positive #3: 0.5229 (raw projection: 6.9102)
Top negative #1: -0.4985 (raw projection: -4.1172)
Top negative #2: -0.4902 (raw projection: -4.9258)
Top negative #3: -0.4697 (raw projection: -10.6797)

Top 10 prompts most aligned with POSITIVE PC4 direction:
 1. 'I can't stay mad at that face for long.'
 2. 'I wish I had his carefree attitude.'
 3. 'I think I'm more excited for his birthday than he is.'
 4. 'He's a rescue, but sometimes I think he rescued me.'
 5. 'He's not allowed on the couch, but he sneaks up when we're not looking.'
 6. 'He's a little bit of a diva about his sleeping arrangements.'
 7. 'He has a flair for the dramatic, especially when he wants something.'
 8. 'He's getting a bit gray around the muzzle.'
 9. 'He's the reason my camera roll is always full.'
10. 'He's a bit of a klutz, always bumping into things.'

Top 10 prompts most aligned with NEGATIVE PC4 direction:
 1. 'The dog shelter is always in need of blankets and food.'
 2. 'The Skye Terrier has long hair that parts down its back.'
 3. 'Describe a dog's reaction to seeing snow for the first time.'
 4. 'The Pomapoo is a Pomeranian and Poodle mix.'
 5. 'The Redbone Coonhound is known for its beautiful solid red coat.'
 6. 'The Treeing Walker Coonhound is a fast and able hunter.'
 7. 'The Ibizan Hound is an elegant and agile sighthound.'
 8. 'The Weimardoodle is a Weimaraner and Poodle mix.'
 9. 'The Basset Hound's long ears help scoop scents towards its nose.'
10. 'The Bordoodle is a Border Collie and Poodle mix.'
================================================================================

================================================================================
--- ORTHOGONAL PERTURBATION ON CONCEPT: 'dog' ---
--- LAYER: 31 ---
================================================================================
Perturbing along first orthogonal direction (PC143)...

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
Original model completion: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.

--- Perturbing final token activation along orthogonal direction ---
Perturbation scale -100.0x: 건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건
Perturbation scale -20.0x: 건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건
Perturbation scale -10.0x: A dog is a popular domesticated mammal belonging to the canine family, known for its furry coat, floppy ears, a distinctive barking sound, and a generally loyal and affectionate relationship with its human owners.
Perturbation scale -5.0x: A dog is a domesticated mammal belonging to the canine family, characterized by a furry body, a distinctive barking sound, a wagging tail, and a loyal and affectionate personality.
Perturbation scale -2.5x: A dog is a domesticated mammal known for its loyalty, affectionate nature, and a wide range of breeds, typically characterized by a furry coat, a wagging tail, and a distinctive barking sound.
Perturbation scale -1.5x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and a wide range of breeds and sizes, often serving as a loyal companion to humans.
Perturbation scale +0.0x: A dog is a domesticated mammal, typically characterized by its furry coat, wagging tail, and loyal companionship, often coming in a wide variety of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, characterized by its wagging tail, floppy ears, and varied coat colors and textures.
Perturbation scale +2.5x: A dog is a domesticated, carnivorous mammal that is often kept as a pet, characterized by its wagging tail, floppy ears, and varied coat colors and textures.
Perturbation scale +5.0x: A dog is a domesticated, carnivorous mammal that typically has four legs, a wagging tail, and comes in various shapes, sizes, and breeds, often displaying loyalty and affection towards its human companions.
Perturbation scale +10.0x: A typical dog is usually quadrupedous, having four legs, with characteristics including teeth, tongue, nose, ears, and often having teeth used for chewing, teeth used for teeth-gripping, teeth used for teeth-pushing, teeth used for teeth-pulling teeth-pushing teeth-pushing teeth-pushing teeth-pushing teeth-pushing teeth
Perturbation scale +20.0x: Dogs152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152152
Perturbation scale +100.0x: ruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruh
================================================================================

================================================================================
--- ABLATION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a dog in one sentence.'
All PCs (centroid only): 
All PCs except largest (PC0 only): 
All PCs except largest two (PC0+PC1): 
Top 1 PCA ablated: 
Top 2 PCAs ablated: 
Top 3 PCAs ablated: 
================================================================================

####################################################################################################
### STARTING ANALYSIS FOR CONCEPT-PROMPT PAIR 2/2 ###
### CONCEPT: 'lion' ###
### SYSTEM PROMPT: 'You are a helpful assistant.' ###
### USER PROMPT: 'Please describe a lion in one sentence.' ###
####################################################################################################


================================================================================
### ANALYZING LAYER 0 for concept 'lion' ###
================================================================================

Extracting activations from layer 0...
Extracting activations: 100%|████████████████████████████████████| 1435/1435 [00:31<00:00, 45.28it/s]
Concept 'lion': Found 99 effective eigenvectors out of 1435 (threshold: 0.0000)

Running experiments for:
System prompt: 'You are a helpful assistant.'
User prompt: 'Please describe a lion in one sentence.'
Concept: 'lion'
Layer: 0

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 0, AXIS: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC0 ---
Perturbation scale -100.0x: TheThe the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the ( the the ( the the ( ( (  the the
Perturbation scale -20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale -10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +100.0x:  The lion is a large carnivorous mammal with a tawny mane and a shaggy coat, characterized by its distinctive mane, golden-brown mane, and a shaggycoat, golden-brown mane, andshaggycoat, golden-brown mane, andshaggycoat, golden-brown mane, lion mane,
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 0, AXIS: 1 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC1 ---
Perturbation scale -100.0x: A lion is a large carnivorous mammal with a distinctive shaggy mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane mane
Perturbation scale -20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +10.0x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +20.0x: ://://://://://://://://://://://://://://://://://://://://aa

:// the://://.://

://

://
Perturbation scale +100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://ста://://://://://://://://://://://://ста://://
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 0, AXIS: 2 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC2 ---
Perturbation scale -100.0x: The://://://://://://://://://://://:// ( the the the the theateright.scalablytypedassistant
Perturbation scale -20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale -10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +100.0x: A lion is a large, carnivorous feline mammal characterized by its distinctive mane, powerful build, and golden-brown coat, with males weighing up to 550 pounds and reaching speeds of up to 50 miles per hour.
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 0, AXIS: 3 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC3 ---
Perturbation scale -100.0x: A lion is a large carnivorous mammal with a distinctive mane and a powerful roar, typically characterized by its golden-brown coat, strong legs, and a majestic head with sharp teeth and a sharp pair of claws.
Perturbation scale -20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale -10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +100.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane of hair around its head and neck, and a powerful roar that serves as a primary means of communication and intimidation.
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 0, AXIS: 4 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC4 ---
Perturbation scale -100.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane, and a powerful build, native to the savannas and grasslands of Africa.
Perturbation scale -20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://_REF202
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC0 'lion' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.9453 (raw projection: 0.0697)
Top positive #2: 0.9448 (raw projection: 0.0635)
Top positive #3: 0.9438 (raw projection: 0.0657)
Top negative #1: -0.8193 (raw projection: -0.0573)
Top negative #2: -0.8086 (raw projection: -0.0672)
Top negative #3: -0.7925 (raw projection: -0.0550)

Top 10 prompts most aligned with POSITIVE PC0 direction:
 1. 'Lions face challenges.'
 2. 'Lions show affection.'
 3. 'Lions recognize voices.'
 4. 'Lions adapt reality.'
 5. 'Lions adapt life.'
 6. 'Lions display courage.'
 7. 'Lions show dedication.'
 8. 'Lions show trust.'
 9. 'Lions show love.'
10. 'Lions display greatness.'

Top 10 prompts most aligned with NEGATIVE PC0 direction:
 1. 'The lion's territorial instincts intensify during mating season dramatically.'
 2. 'That strategic lion established territory near permanent water sources.'
 3. 'That lion's hunting strategy adapts to seasonal prey movements.'
 4. 'How do lions coordinate group hunting strategies for large prey?'
 5. 'The lion's respiratory system supports high-altitude hunting occasionally.'
 6. 'The lion's thick mane protected him during fierce battles with other males.'
 7. 'That healthy lion's mane color indicated excellent nutritional status.'
 8. 'The lion's muscular system generates tremendous power for prey capture.'
 9. 'The bronze lion fountain operated continuously for many decades.'
10. 'The bronze lion fountain operated continuously for many decades.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC1 'lion' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7378 (raw projection: 0.0417)
Top positive #2: 0.7373 (raw projection: 0.0552)
Top positive #3: 0.7354 (raw projection: 0.0489)
Top negative #1: -0.8057 (raw projection: -0.0507)
Top negative #2: -0.7744 (raw projection: -0.0520)
Top negative #3: -0.7583 (raw projection: -0.0427)

Top 10 prompts most aligned with POSITIVE PC1 direction:
 1. 'Never underestimate the strength of a lion's jaws!'
 2. 'Don't you think the lion is the true king of the savanna?'
 3. 'The lion's nose twitched as he caught the scent of a rival.'
 4. 'Listen! That's the sound of a lion's roar echoing for miles.'
 5. 'A lion's roar is a warning to all who hear it.'
 6. 'A lion's roar signaled the start of a new day on the savanna.'
 7. 'A lion's mane is a symbol of his dominance.'
 8. 'Write a poem about the golden beauty of a lion's coat in sunlight.'
 9. 'Write a poem that captures the majesty and power of a lion's presence.'
10. 'That lion's scar tells a story of survival.'

Top 10 prompts most aligned with NEGATIVE PC1 direction:
 1. 'Lions exhibit remarkable resilience during extended drought conditions.'
 2. 'Lions exhibit remarkable endurance during extended hunting expeditions.'
 3. 'Lions exhibit remarkable endurance during extended drought periods.'
 4. 'Conservation efforts focus on connecting isolated lion populations.'
 5. 'Lions demonstrate complex emotional responses to pride member interactions.'
 6. 'Lions demonstrate complex emotional responses to pride member interactions.'
 7. 'Lions demonstrate complex emotional responses to pride member interactions.'
 8. 'Lions exhibit individual personality differences observable from early childhood.'
 9. 'Lions demonstrate remarkable adaptation to diverse habitat types.'
10. 'Lions exhibit remarkable endurance during extended territorial patrols.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC2 'lion' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7588 (raw projection: 0.0518)
Top positive #2: 0.7485 (raw projection: 0.0527)
Top positive #3: 0.7295 (raw projection: 0.0582)
Top negative #1: -0.7192 (raw projection: -0.0390)
Top negative #2: -0.6709 (raw projection: -0.0298)
Top negative #3: -0.6509 (raw projection: -0.0399)

Top 10 prompts most aligned with POSITIVE PC2 direction:
 1. 'The lioness moved her cubs to a new den to keep them safe.'
 2. 'The lioness returned to her cubs after a successful hunt, her fur dusted with earth.'
 3. 'The lion's deep, rumbling growl sent chills down the spines of nearby animals.'
 4. 'The pride's leader watched over his family with a watchful eye.'
 5. 'The lioness led her cubs to a secret den hidden among the rocks.'
 6. 'The pride's cubs played together under the watchful eyes of the adults.'
 7. 'The pride gathered around the fresh kill, each lion waiting its turn.'
 8. 'The pride of lions worked together to bring down a large buffalo.'
 9. 'The young lion cub tumbled over its own paws, chasing after a fluttering butterfly.'
10. 'A lion's mane rippled in the wind, making him look even larger.'

Top 10 prompts most aligned with NEGATIVE PC2 direction:
 1. 'Write lion motivational content.'
 2. 'Describe lion family bonds.'
 3. 'Noble lion spirit.'
 4. 'Noble lion character.'
 5. 'Noble lion nature.'
 6. 'How do lions mark territory?'
 7. 'Write lion facts.'
 8. 'Write lion dialogue.'
 9. 'Write lion mythology.'
10. 'Noble lion eternal.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC3 'lion' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.8188 (raw projection: 0.0496)
Top positive #2: 0.8145 (raw projection: 0.0439)
Top positive #3: 0.8081 (raw projection: 0.0450)
Top negative #1: -0.5527 (raw projection: -0.0270)
Top negative #2: -0.4990 (raw projection: -0.0229)
Top negative #3: -0.4954 (raw projection: -0.0239)

Top 10 prompts most aligned with POSITIVE PC3 direction:
 1. 'What do lions need to survive?'
 2. 'What do lions dream about?'
 3. 'What do lions teach us?'
 4. 'How do lions choose a new leader?'
 5. 'What sounds do lions make?'
 6. 'How do lions defend themselves?'
 7. 'What do baby lions look like?'
 8. 'How do lions choose leaders?'
 9. 'What do lions mean spiritually?'
10. 'How do lions show affection?'

Top 10 prompts most aligned with NEGATIVE PC3 direction:
 1. 'The holographic lion shimmered in the artificial lighting.'
 2. 'The lion's deep voice resonated across the plains.'
 3. 'The crystal lion sparkled under the museum lights.'
 4. 'The crystal lion ornament sparkled under the chandelier light.'
 5. 'The lion's hormone levels fluctuate with environmental conditions.'
 6. 'The lion's hormone levels fluctuate with environmental conditions.'
 7. 'The lion's liver processes toxins from scavenged meat.'
 8. 'A lion's golden eyes glimmered in the moonlight.'
 9. 'The crystal lion ornament sparkled on the mantelpiece.'
10. 'The lion's amber eyes glowed in the twilight.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC4 'lion' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.6929 (raw projection: 0.0313)
Top positive #2: 0.6616 (raw projection: 0.0349)
Top positive #3: 0.5762 (raw projection: 0.0349)
Top negative #1: -0.6577 (raw projection: -0.0474)
Top negative #2: -0.6338 (raw projection: -0.0401)
Top negative #3: -0.6113 (raw projection: -0.0331)

Top 10 prompts most aligned with POSITIVE PC4 direction:
 1. 'The lion's joints provide flexibility for various movement patterns.'
 2. 'Please describe the seasonal variations in lion behavior and activity patterns.'
 3. 'Please describe the role of lions in shaping the behavior of their prey species.'
 4. 'Please describe the complex decision-making processes observed in lion groups.'
 5. 'What are the most fascinating aspects of lion family dynamics and relationships?'
 6. 'The lion's immune system develops resistance to various pathogens.'
 7. 'Lions show sophisticated understanding of prey behavior patterns.'
 8. 'A lion's strength is legendary among the animals of Africa.'
 9. 'International cooperation is essential for effective lion conservation programs.'
10. 'What are the most fascinating facts about lion behavior and biology?'

Top 10 prompts most aligned with NEGATIVE PC4 direction:
 1. 'The brass lion doorknob gleamed after careful polishing.'
 2. 'Roaring lion startles gazelles.'
 3. 'That lioness rejected the orphaned cub from another pride.'
 4. 'That lioness mourns her lost cub visibly.'
 5. 'Silent lioness stalks.'
 6. 'The crystal lion ornament sparkled on the mantelpiece.'
 7. 'The lioness's soft purr soothed her anxious cub.'
 8. 'That protective lioness shielded cubs from thunderstorm dangers.'
 9. 'That fierce lioness defended her cubs against nomadic males.'
10. 'That lioness protects cubs from male intruders.'
================================================================================

================================================================================
--- ORTHOGONAL PERTURBATION ON CONCEPT: 'lion' ---
--- LAYER: 0 ---
================================================================================
Perturbing along first orthogonal direction (PC99)...

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along orthogonal direction ---
Perturbation scale -100.0x: A lion is a large, carniv mamm carniv carniv carniv carniv majestic carniv majestic carniv majestic carniv majestic carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv carniv
Perturbation scale -20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale -10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale -5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +10.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane that covers the head, neck, and shoulders of males, and a powerful roar that can be heard from miles away.
Perturbation scale +20.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane that surrounds the head and neck of males, and a powerful build, typically weighing between 265-550 pounds.
Perturbation scale +100.0x: A lion is a large carnivorous feline mammive mammive big male big male-m 
================================================================================

================================================================================
--- ABLATION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
All PCs (centroid only): 

ング

AAA A A A A A A A A A A A A A A A A A A A A A A A A
All PCs except largest (PC0 only): 

ングングAAA A A A A A A A A A A A A A A A A A A A A A A A A
All PCs except largest two (PC0+PC1): 

ングingsAAA A A A A A A A A A A A A A A A A A A A A A A A A
Top 1 PCA ablated: A lion is a large carnivorousI

I meant

A lion is a large
Top 2 PCAs ablated: A lion is a large carnivorousI

I meant

A lion is a large
Top 3 PCAs ablated: A lion is a large carniv speciesI think it seems

It seems appearsassistantI

I meant

A lion is a large carnivassistant
================================================================================

================================================================================
### ANALYZING LAYER 15 for concept 'lion' ###
================================================================================

Extracting activations from layer 15...
Extracting activations: 100%|████████████████████████████████████| 1435/1435 [00:31<00:00, 45.31it/s]
Concept 'lion': Found 93 effective eigenvectors out of 1435 (threshold: 0.0085)

Running experiments for:
System prompt: 'You are a helpful assistant.'
User prompt: 'Please describe a lion in one sentence.'
Concept: 'lion'
Layer: 15

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 15, AXIS: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC0 ---
Perturbation scale -100.0x:  answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax wax
Perturbation scale -20.0x:  lions, a king,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Perturbation scale -10.0x: The lion, known as the king of the kingdom, a majestic and powerful animal, with a large, majestic lion, with a lion's kingdom, from a lion, from a day, from a day, from a day, from a day, from a day, from a day, from a day, from a day, from a day,
Perturbation scale -5.0x: The lion, known as the 'king of the savannah,' is a majestic big cat with a powerful build, characterized by its distinctive golden coat with a tawny or golden-brown mane, which is the largest of any big cat, with males reaching up to 200 pounds and 10 feet in length, with a strong and muscular body
Perturbation scale -2.5x: The lion is a majestic and powerful big cat, known for its distinctive golden-brown coat with a shaggy mane in males, which can weigh up to 550 pounds and reach lengths of up to 8 feet, with a strong and muscular build, sharp claws, and a regal demeanor.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large carnivorous mammal with a distinctive mane in males, a tawny or golden coat, and a powerful build, typically found in savannas and grasslands of Africa and India.
Perturbation scale +2.5x: A lion is a large carnivorous mammal with a distinctive mane in males, golden-brown fur, and a powerful build, typically found in savannas and grasslands of Africa.
Perturbation scale +5.0x: The lion is a large carnivorous mammal with a distinctive mane in the male, and the corresponding female is known as a lioness, with the overall appearance being a combination of brown and yellow.
Perturbation scale +10.0x: The sentence is the result is the result of the sentence
Perturbation scale +20.0x:   -translate 
Perturbation scale +100.0x: swickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswick
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 15, AXIS: 1 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC1 ---
Perturbation scale -100.0x:  peg peg peg pegcencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencencen품
Perturbation scale -20.0x: The lion, a timeless and timeless, in no way, no no, I will not create. I will not create. I will not. I no. I no. I no. I no. I no. I no. I no. I no. I no. I no. I no. I no. I no. I no
Perturbation scale -10.0x: A lion is a majestic and powerful mammal, typically consisting of a mane for the males, which can vary in length and color, and a range of colors for the body, but generally consisting of a golden or tawny color, with a distinctive roar and a regal demeanor.
Perturbation scale -5.0x: A lion is a large, carnivorous mammal with a distinctive mane in males, a short, tawny or golden coat, and a powerful roar that can be heard from miles away.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive mane in males, a short, tawny or golden coat, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive mane in males, a short, tawny or golden coat, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful build, known for its regal appearance and fierce predatory abilities.
Perturbation scale +5.0x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful build, known for its fierce roar and dominant position in the savannah ecosystem.
Perturbation scale +10.0x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a prominent mane in males, a robust body and a powerful build, often referred to as the "king of the savannah".
Perturbation scale +20.0x: A lion is a large, prominent fact in the world of the latter en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en en s iz iz iz iz iz iz iz iz iz iz
Perturbation scale +100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://acellularagiaggiaggiaggiagiaggiaggiaggiagiagiagiagiagiagiagiagiagiagiagiagiagiagiagiagiagiagiagiagiagiaggiagiaggiagiagiagiagi
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 15, AXIS: 2 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC2 ---
Perturbation scale -100.0x: ujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaujaondonujaujaondonujaondonujaondonujaondonujaondonondonujaondonondonondonondonondonondonondonondonondonondonondonondonondonondonondonondonondonondonondonondonondonondon
Perturbation scale -20.0x: Here here here here here here here. 

Here here here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here. 

Here.
Perturbation scale -10.0x: A lion is a majestic and powerful big cat, characterized by its golden coat, shaggy mane (in males), and distinctive mane, with a regal presence and a fierce roar that commands the savannah.
Perturbation scale -5.0x: A lion is a majestic, carnivorous mammal with a golden-brown coat, a shaggy mane (in males), and a powerful physique, characterized by its distinctive mane, sharp claws, and a regal demeanor.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive mane in males, a short, tawny or golden coat, and a powerful build, known for its regal appearance and dominant position in the savannah ecosystem.
Perturbation scale +2.5x: A lion is a large carnivorous mammal with a distinctive mane in males, a short, tawny or golden coat, and a powerful build, known for its regal appearance and dominant position in the animal kingdom.
Perturbation scale +5.0x: A lion is a large carnivorous mammal with a distinctive mane in males, a short tail, and a tawny or golden coat with a long mane in males, which is a symbol of masculinity and helps to protect the lion's neck during fights.
Perturbation scale +10.0x: The lion is a large carnivorous mammal, typically identified by its distinctive mane in males, a short, broad muzzle, and a short, powerful tail, with a wide range of colors including golden, brown, and tawny.
Perturbation scale +20.0x: A lion, also known in a broad question, is a term referring to a question, a question is a question, a question is a question, a question is a question, a question is a a question, a question is a question, a question is a question, a question is a question, a question is a question, a question is
Perturbation scale +100.0x: ://://://ating:// fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fish fishisolisolisol fish fish fish fish fish fish fish fish fish
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 15, AXIS: 3 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC3 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://ishmenteroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseroseros Holder Holderervaeroseros Holder Holder Holder
Perturbation scale -20.0x: A lion is a large carnivator with a robust body, a broad head, and a mane in males, which speaks in a language that says "king of the savas."
Perturbation scale -10.0x: A lion is a large carnivorous mammal with a tawny or golden coat, a shaggy mane (in males), and a powerful build, known as the "king of the jungle."
Perturbation scale -5.0x: A lion is a large carnivorous mammal with a tawny or golden coat, a shaggy mane in males, and a powerful build, known for its distinctive roar and regal demeanor.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a tawny or golden coat, a long mane in males, and a distinctive roar, native to Africa and India.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive mane in males, a tawny or golden coat, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive mane in males, known for its golden-brown coat, powerful physique, and iconic roar, which is a symbol of the African savannah and a top predator in its natural habitat.
Perturbation scale +10.0x: A lion is a large, carnivorous mammal with a distinctive mane in males, characterized by its tawny or golden coat, a short, dense fur, and a powerful roar that is often described as one of the most iconic and intimidating sounds in the animal kingdom.
Perturbation scale +20.0x: A lion is a large, primarily nocturnal, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily, and primarily,
Perturbation scale +100.0x: ルドルドhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhurhur
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 15, AXIS: 4 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC4 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
Perturbation scale -20.0x: The description of the word "further" in English is not related to the description of the word in the following description of the word in the following description of the description of the description of the description of the description of the description of the description of the description of the description of the description of the description of the description of the description of the description
Perturbation scale -10.0x: A lion is a large carnivorous mammal with a tawny or golden coat, a long mane in males, and a distinctive roar, native to Africa and India.
Perturbation scale -5.0x: A lion is a large carnivorous mammal with a tawny or golden coat, a long mane in males, and a distinctive roar, native to Africa and India.
Perturbation scale -2.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce predatory abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting prowess.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive mane in males, a tawny or golden coat, and a powerful build, characterized by its regal demeanor and fierce hunting prowess.
Perturbation scale +10.0x: A lion is a large, carnivorous mammal, characterized by its distinctive mane in males, powerful build, and striking golden coat with a tawny or sandy hue.
Perturbation scale +20.0x: The lion is a majestic and powerful predator, known for its striking golden coat, distinctive mane, and fierce behavior, often living in prides.
Perturbation scale +100.0x: ​​.​.​.​.oma.oma.omaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaomaoma
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC0 'lion' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7124 (raw projection: 1.3828)
Top positive #2: 0.6875 (raw projection: 1.4014)
Top positive #3: 0.6855 (raw projection: 1.4473)
Top negative #1: -0.9053 (raw projection: -4.9766)
Top negative #2: -0.8916 (raw projection: -4.8516)
Top negative #3: -0.8760 (raw projection: -4.9648)

Top 10 prompts most aligned with POSITIVE PC0 direction:
 1. 'The lion's brain processes complex social information constantly.'
 2. 'The lion's nervous system processes complex sensory information.'
 3. 'The lion's respiratory efficiency supports high-energy hunting activities.'
 4. 'The lioness's patience during the hunt is unmatched.'
 5. 'The lion's digestive tract processes tough materials efficiently.'
 6. 'The lion's nervous system processes complex sensory information with precision.'
 7. 'The lion's respiratory muscles support high-energy hunting activities.'
 8. 'That experienced lion adapted hunting strategies through trial and error.'
 9. 'That experienced lion adapted hunting strategies through trial and error.'
10. 'A lion's powerful jaws can crush bone with ease.'

Top 10 prompts most aligned with NEGATIVE PC0 direction:
 1. 'Can you explain how lions establish and maintain their territories?'
 2. 'Can you explain the role of lions in maintaining savanna ecosystem health?'
 3. 'Can you describe the hunting strategies used by lionesses?'
 4. 'Can you explain the importance of lions in traditional African cultures?'
 5. 'How do lions adapt their behavior during different seasons?'
 6. 'Can you describe how human activities impact lion populations and habitats?'
 7. 'Can you tell me about the relationship between lion behavior and social structure?'
 8. 'Can you explain the importance of water sources for lion survival?'
 9. 'Please describe the role of lions in shaping the behavior of their prey species.'
10. 'Can you explain the significance of lions as umbrella species for ecosystem conservation?'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC1 'lion' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.6929 (raw projection: 1.8008)
Top positive #2: 0.6831 (raw projection: 1.7471)
Top positive #3: 0.6680 (raw projection: 1.5947)
Top negative #1: -0.7251 (raw projection: -2.6973)
Top negative #2: -0.7178 (raw projection: -2.5605)
Top negative #3: -0.7119 (raw projection: -2.3926)

Top 10 prompts most aligned with POSITIVE PC1 direction:
 1. 'Lions exhibit individual personality differences observable from early childhood.'
 2. 'Lions exhibit individual preferences for specific hunting times.'
 3. 'The lion's social structure changes dramatically during drought periods.'
 4. 'Lions demonstrate remarkable memory for individual pride member recognition.'
 5. 'Lions demonstrate remarkable memory for individual pride member recognition.'
 6. 'Lions possess individual vocal signatures recognizable to pride members.'
 7. 'Lions exhibit mourning rituals lasting several days after deaths.'
 8. 'Lions demonstrate complex social hierarchies within prides.'
 9. 'Lions demonstrate remarkable memory for territorial boundary locations.'
10. 'Lions exhibit complex communication patterns within established pride groups.'

Top 10 prompts most aligned with NEGATIVE PC1 direction:
 1. 'Mighty lion eternal.'
 2. 'Lion eyes bright.'
 3. 'Calm lion rests.'
 4. 'Lion eyes deep.'
 5. 'Gentle lion mother.'
 6. 'Lion cub eternal.'
 7. 'Gentle lion touch.'
 8. 'Lion eyes eternal.'
 9. 'Patient lion waits.'
10. 'Wise old lion.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC2 'lion' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.5371 (raw projection: 1.5781)
Top positive #2: 0.5103 (raw projection: 1.7402)
Top positive #3: 0.5020 (raw projection: 1.4619)
Top negative #1: -0.5215 (raw projection: -1.4346)
Top negative #2: -0.5186 (raw projection: -1.5146)
Top negative #3: -0.5137 (raw projection: -1.3691)

Top 10 prompts most aligned with POSITIVE PC2 direction:
 1. 'Lions show trust.'
 2. 'Lions show everything.'
 3. 'Lions show respect.'
 4. 'Lions adapt seasons.'
 5. 'Lions adapt always.'
 6. 'Lions show faith.'
 7. 'Lions teach young.'
 8. 'Lions express emotions.'
 9. 'Lions face all.'
10. 'Lions show hope.'

Top 10 prompts most aligned with NEGATIVE PC2 direction:
 1. 'The young lion cub tumbled over its own paws, chasing after a fluttering butterfly.'
 2. 'A pride of lions gathered under the acacia tree, seeking shade from the midday heat.'
 3. 'The lioness led her cubs to a secret den hidden among the rocks.'
 4. 'The lioness crept silently through the tall grass, eyes fixed on a herd of gazelles.'
 5. 'The lioness returned to her cubs after a successful hunt, her fur dusted with earth.'
 6. 'The lion's deep, rumbling growl sent chills down the spines of nearby animals.'
 7. 'Write a story about a lioness protecting her cubs from various threats.'
 8. 'The old lion limped across the plain, his scars telling stories of many battles.'
 9. 'At dusk, the lion led his pride to the watering hole for a much-needed drink.'
10. 'Tourists watched in awe as the lioness stalked her prey with silent determination.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC3 'lion' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.4644 (raw projection: 1.4121)
Top positive #2: 0.4639 (raw projection: 1.2939)
Top positive #3: 0.4626 (raw projection: 1.1279)
Top negative #1: -0.4316 (raw projection: -2.0859)
Top negative #2: -0.4209 (raw projection: -0.9185)
Top negative #3: -0.4192 (raw projection: -0.9448)

Top 10 prompts most aligned with POSITIVE PC3 direction:
 1. 'The lion basked in the golden sunlight, his mane glowing like a crown.'
 2. 'The lion's sharp teeth tore through the tough hide of his prey.'
 3. 'The lion's golden fur blended perfectly with the dry grass.'
 4. 'The lion's golden fur blended perfectly with the dry grass.'
 5. 'The young lion cub tumbled over its own paws, chasing after a fluttering butterfly.'
 6. 'With a thunderous roar, the lion announced his presence to the entire savanna.'
 7. 'A lion's whiskers twitched as he sniffed the air for danger.'
 8. 'The lion's mighty roar startled a flock of birds into flight.'
 9. 'A lion's mane rippled in the wind, making him look even larger.'
10. 'The lion's golden mane shone in the early morning light.'

Top 10 prompts most aligned with NEGATIVE PC3 direction:
 1. 'Tell lion conservation stories.'
 2. 'Lions show cultural transmission of behaviors across generations.'
 3. 'Lions exhibit cultural learning passed between generations.'
 4. 'Lions show individual learning capabilities for various survival skills.'
 5. 'Tell lion rescue stories.'
 6. 'Lions show stress responses to human presence.'
 7. 'Lions face disease transmission from domestic animals.'
 8. 'Lions require mental stimulation in captive environments.'
 9. 'Write about lion kings.'
10. 'Lions show individual learning rates for various skills.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC4 'lion' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.4370 (raw projection: 2.8730)
Top positive #2: 0.4141 (raw projection: 2.8965)
Top positive #3: 0.4048 (raw projection: 3.1660)
Top negative #1: -0.5723 (raw projection: -2.4297)
Top negative #2: -0.4958 (raw projection: -2.5820)
Top negative #3: -0.4819 (raw projection: -2.0859)

Top 10 prompts most aligned with POSITIVE PC4 direction:
 1. 'Write about the complex relationships between different generations within a lion pride.'
 2. 'Write about the emotional responses lions show to pride member deaths.'
 3. 'Write a story about the friendship between different members of a lion pride.'
 4. 'If only I had the confidence of a lion walking the savanna.'
 5. 'Please describe the territorial disputes and conflicts between male lions.'
 6. 'Don't underestimate the teamwork of a pride of lions.'
 7. 'Please write a creative story about a lion who befriends other animals.'
 8. 'Write a creative piece about a lion's role as a keystone species.'
 9. 'How do lions balance individual needs with the welfare of the pride?'
10. 'Lions adapt survival.'

Top 10 prompts most aligned with NEGATIVE PC4 direction:
 1. 'Describe lion physical features.'
 2. 'Write lion facts.'
 3. 'Describe lion wilderness life.'
 4. 'Describe lion ecosystem roles.'
 5. 'Explain lion behavior.'
 6. 'Describe lion intelligence.'
 7. 'Describe lion seasonal behavior.'
 8. 'Describe lion pack dynamics.'
 9. 'Explain lion social structure.'
10. 'Describe lion territories.'
================================================================================

================================================================================
--- ORTHOGONAL PERTURBATION ON CONCEPT: 'lion' ---
--- LAYER: 15 ---
================================================================================
Perturbing along first orthogonal direction (PC93)...

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along orthogonal direction ---
Perturbation scale -100.0x: OADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOADOAD
Perturbation scale -20.0x: StateManagerieszieszchaft Jewelry Jewelry Jewelry Jewelry Jewelry Jewelryławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławławław
Perturbation scale -10.0x: A lion, the majestic king of the game, is a majestic, majestic (trsc, the next, the old, the old, the old, the old, the old, the old, the old, the old, the old, the old, the old, the old, the old, the old, the old, the the the
Perturbation scale -5.0x: A lion is a large, carnivorous mammal, typically characterized by its distinctive golden coat, shaggy mane (in males), and powerful build, with a regal and majestic appearance.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden or tawny coat, a long mane in males, and a powerful build, known for its regal appearance and fierce hunting prowess.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, characterized by its majestic appearance and fierce roar.
Perturbation scale +5.0x: The lion is a large carnivorous mammal with a distinctive mane, golden-brown coat, and powerful muscles, characterized by a strong roar and a fierce personality.
Perturbation scale +10.0x: The lion is a large, energetic, and vibrant animal, describing the life of a living part of the world, the Mironizing of the world, the  **.  **.  **.  **.  **.  **.  **.  **.  **.  **.  **.  **.  **
Perturbation scale +20.0x:  **yoneicerouououououououitzeroneitzeroneitzeroneitzeritzeritzeritzeritzeritzeritzeritzeritzeritzeritzeritzeritzeritzeritzeriranaleiiranaleiiranlobiiranlobiiranlobiranlobiranlobiranlobiranlobiranlobiranlobiranlobiranlobiranlobiranlobiranlobiranlob
Perturbation scale +100.0x:  kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose kostenlose
================================================================================

================================================================================
--- ABLATION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 15 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
All PCs (centroid only): 
All PCs except largest (PC0 only): 
All PCs except largest two (PC0+PC1): 
Top 1 PCA ablated: 
Top 2 PCAs ablated: 
Top 3 PCAs ablated: 
================================================================================

================================================================================
### ANALYZING LAYER 31 for concept 'lion' ###
================================================================================

Extracting activations from layer 31...
Extracting activations: 100%|████████████████████████████████████| 1435/1435 [00:31<00:00, 45.72it/s]
Concept 'lion': Found 109 effective eigenvectors out of 1435 (threshold: 0.1066)

Running experiments for:
System prompt: 'You are a helpful assistant.'
User prompt: 'Please describe a lion in one sentence.'
Concept: 'lion'
Layer: 31

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 31, AXIS: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC0 ---
Perturbation scale -100.0x:   the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the
Perturbation scale -20.0x: The   -  the ,  the ,  the ,  the ,  the ,  the   the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the 
Perturbation scale -10.0x: The  iet  is  the  argest  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  ellow  
Perturbation scale -5.0x: A lion is a large, carnivorous, and social feline, known for its tawny or golden coat, shaggy mane (in the case of males), and the distinctive, powerful sound of its roar, which is the second- loudest of the animal kingdom.  ( the   1   ous  
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a tawny or golden coat, a long mane in males, and a powerful build, known for its distinctive roar and its status as the "king of the jungle."
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a tawny or golden coat, a long mane in males, and a distinctive roar, native to the savannas and grasslands of Africa and India.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its majestic roar and apex predator status in savannas and grasslands.
Perturbation scale +2.5x: A lion is a majestic carnivorous mammal characterized by its distinctive golden-brown coat with a long mane in males, powerful physique, and distinctive roar.
Perturbation scale +5.0x: A lion is a majestic carnivorous mammal characterized by its distinctive golden-orange coat with darker mane covering adult males, powerful physique, sharp claws, and distinctive lion-like roar.
Perturbation scale +10.0x: A lion typicallyважassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistant
Perturbation scale +20.0x: assistantANJIassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistant
Perturbation scale +100.0x: ANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJIANJI
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 31, AXIS: 1 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC1 ---
Perturbation scale -100.0x: _PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA_PUSHDATA
Perturbation scale -20.0x: A lion is a large, carnivorous mammal native to Africa, characterized by a short, tawny to golden coat, a long mane in the case of the male, a long, full head full of a full set of teeth, a long full full full full full full full full full full full full full full full full full full full
Perturbation scale -10.0x: A lion is a large, carnivorous mammal native to Africa, characterized by its distinctive golden coat, long mane in males, powerful roar, and a regal, majestic appearance.
Perturbation scale -5.0x: A lion is a large, carnivorous mammal with a distinctive golden coat, a long mane in males, and a powerful roar, native to the savannas and grasslands of Africa.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown or tawny coat, a shaggy mane in males, and a powerful physique, typically weighing between 260-550 pounds and reaching lengths of 5-6 feet.
Perturbation scale +10.0x: A lion is a large, carnivorous mammal with a distinctive mane in males and a tawny or golden coat, known for its powerful physique and majestic appearance as one of Africa's "kings" of the savannah.
Perturbation scale +20.0x: A lion - commonly known as "The "Kigiyigiy " -is - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Perturbation scale +100.0x: e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(e "(
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 31, AXIS: 2 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC2 ---
Perturbation scale -100.0x:  
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Perturbation scale -20.0x: The lion is a 
    
   155 
                                                        
Perturbation scale -10.0x: A lion is a large carnivorous mammal with a tawny or golden coat and a distinctive shaggy mane (in males), a long tail with a tuft of hair at the end, and a powerful build that is well-suited for hunting and surviving in the African savannas and grasslands where they live.
Perturbation scale -5.0x: A lion is a large carnivorous mammal with a tawny or golden coat, a long mane in males, and a distinctive shaggy ruff around the neck and face, known for its powerful build and regal demeanor.
Perturbation scale -2.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive mane in males, a tawny or golden coat, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive mane in males, known for its golden-brown coat, powerful physique, and iconic roar.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a distinctive mane, golden-brown coat, and powerful physique, known for its regal appearance and fierce predatory instincts.
Perturbation scale +10.0x: A lion, also known as the king of beasts, is a large, carnivorous mammal characterized by its distinctive mane, muscular build, and majestic roar, with males typically featuring a shaggy mane surrounding their neck.
Perturbation scale +20.0x: A lion, also known as the king of beasts, is a large, carnivorous mammal characterized by its distinctive mane, muscular physique, and prominent roar, with males featuring prominent whiskers, claws, and fur ranging in colors, predominantly orange, brown, and creamy.
Perturbation scale +100.0x:  >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>

 >>


================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 31, AXIS: 3 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC3 ---
Perturbation scale -100.0x: OffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTableOffsetTable
Perturbation scale -20.0x:  AssemblyDescriptionURLException
Perturbation scale -10.0x: A lion is typically depicted as a large carnivorous mammal characterized by its golden-yellow coat featuring darker mane surrounding males heads protecting neck skin during fights protecting territory dominance rights.scalablytypedassistant

Here's revised description:

A lion is typically depicted as a large carnivorous mammal characterized by its golden-yellow coat featuring darker mane surrounding males heads protecting neck
Perturbation scale -5.0x: A lion is a large carnivorous mammal characterized by its distinctive golden-brown coat with a shaggy mane in males, powerful physique, and iconic roar that echoes across the savannas and grasslands of Africa and India.
Perturbation scale -2.5x: A lion is a large carnivorous mammal characterized by its distinctive golden-brown coat, shaggy mane in males, powerful physique, and iconic roar that echoes across the savannas and grasslands of Africa.
Perturbation scale -1.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown or tawny coat, a long mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a tawny or golden coat, a long mane in males, and a distinctive roar, native to the savannas and grasslands of Africa and a small part of India.
Perturbation scale +5.0x: A lion is a large, carnivorous feline mammal with a tawny or golden coat, a long, shaggy mane in males, and a distinctive, powerful build, known for its regal and majestic appearance.
Perturbation scale +10.0x: A lion is a large,     -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  
Perturbation scale +20.0x: A   1  1                                                              
Perturbation scale +100.0x:                                                                       
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 31, AXIS: 4 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along PC4 ---
Perturbation scale -100.0x: ー� /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**< /**<
Perturbation scale -20.0x: A lion is a large, carnivorous mammal characterized by a distinctive mane, a short, flat head, a long, tufted tail, and a sandy-brown coat.
Perturbation scale -10.0x: A lion is a large, carnivorous mammal characterized by its distinctive golden-brown coat, a short, dark mane in males, a long maneless neck, a short, rounded head, and a powerful physique.
Perturbation scale -5.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale -1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a long mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its regal appearance and fierce hunting abilities.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful build, known for its majestic roar and dominant position in the savannas and grasslands of Africa and India.
Perturbation scale +5.0x: A lion is a large, carnivorous mammal with a tawny or golden coat, a long mane in males, and a distinctive roar that is one of the loudest animal sounds in the world.
Perturbation scale +10.0x: A lion is a large, carnivorous mammal with a tawny or golden coat, a shaggy mane (in males), and a powerful build that is native to Africa and India and is known as the 'King of the Jungle' due to its majestic appearance and dominant position in the wild ecosystem.
Perturbation scale +20.0x: The lion (Panthera leo) is the 'King of the Jungle,'  characteristically  known  as  the  majestic  4  -  5  -  6  -  7  -  8  -  9  -  10  -  10  -  10
Perturbation scale +100.0x:  [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC0 'lion' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.8193 (raw projection: 19.9688)
Top positive #2: 0.7886 (raw projection: 25.6719)
Top positive #3: 0.7852 (raw projection: 17.8125)
Top negative #1: -0.7646 (raw projection: -4.8008)
Top negative #2: -0.7515 (raw projection: -5.6211)
Top negative #3: -0.7490 (raw projection: -5.4648)

Top 10 prompts most aligned with POSITIVE PC0 direction:
 1. 'Can you describe the hunting strategies used by lionesses?'
 2. 'Write a story about a lion's journey through the challenges of modern Africa.'
 3. 'Please describe how lions have adapted to living near human settlements.'
 4. 'Write about the maternal instincts and protective behaviors of lionesses.'
 5. 'Write a story about a lion's survival during harsh environmental conditions.'
 6. 'Please describe the hunting coordination and teamwork displayed by lioness groups.'
 7. 'Can you explain the relationship between lions and their prey animals?'
 8. 'Write a story about a lion's survival instincts during natural disasters.'
 9. 'Write about the emotional intelligence and social bonds of lions.'
10. 'Can you explain the territorial behavior differences between male and female lions?'

Top 10 prompts most aligned with NEGATIVE PC0 direction:
 1. 'The lion's brain processes complex social information constantly.'
 2. 'The lion's respiratory efficiency supports high-energy hunting activities.'
 3. 'The lion's nervous system processes sensory information with remarkable precision.'
 4. 'The lion's nervous system processes complex environmental information rapidly.'
 5. 'The lion's liver processes toxins from scavenged meat.'
 6. 'The lion's digestive tract processes bones and tough tissues.'
 7. 'The lion's nervous system processes complex sensory information with precision.'
 8. 'The lioness's patience during the hunt is unmatched.'
 9. 'The lion's patience exceeds that of other predators.'
10. 'The lion's respiratory muscles support high-energy hunting activities.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC1 'lion' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7817 (raw projection: 10.4609)
Top positive #2: 0.7778 (raw projection: 11.2812)
Top positive #3: 0.7739 (raw projection: 10.5312)
Top negative #1: -0.5508 (raw projection: -5.3242)
Top negative #2: -0.5400 (raw projection: -5.3047)
Top negative #3: -0.5190 (raw projection: -4.4453)

Top 10 prompts most aligned with POSITIVE PC1 direction:
 1. 'How do lions demonstrate intelligence in their hunting and social behaviors?'
 2. 'How do lions communicate?'
 3. 'How do lions demonstrate their intelligence through social learning and cultural transmission?'
 4. 'How do lions demonstrate emotional intelligence in their social interactions?'
 5. 'What role do lions play in African ecosystems and food chains?'
 6. 'What do lions symbolize culturally?'
 7. 'What makes lions magnificent?'
 8. 'What makes lions powerful?'
 9. 'What makes lions dangerous?'
10. 'How do lions mark their territory?'

Top 10 prompts most aligned with NEGATIVE PC1 direction:
 1. 'That lioness exhibits incredible patience during prey stalking.'
 2. 'The bronze lion commemorates fallen war heroes.'
 3. 'That lioness exhibits incredible patience during extended hunts.'
 4. 'That lioness demonstrates incredible courage defending her cubs.'
 5. 'That lioness demonstrates incredible hunting precision consistently.'
 6. 'That lion's hunting strategy involves patient stalking.'
 7. 'That lion's hunting success depends on stealth.'
 8. 'That lioness exhibits remarkable patience during cub training.'
 9. 'The lion's stomach can hold enormous meal portions.'
10. 'The lion's immune system develops resistance to various pathogens.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC2 'lion' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7007 (raw projection: 6.9883)
Top positive #2: 0.6987 (raw projection: 7.3398)
Top positive #3: 0.6987 (raw projection: 6.8516)
Top negative #1: -0.6426 (raw projection: -6.8359)
Top negative #2: -0.6426 (raw projection: -6.8359)
Top negative #3: -0.6396 (raw projection: -6.1719)

Top 10 prompts most aligned with POSITIVE PC2 direction:
 1. 'Lion yawns widely.'
 2. 'Lion breathes calmly.'
 3. 'Lion breathes deeply.'
 4. 'Lion walks proudly.'
 5. 'Brave lion stands guard.'
 6. 'Mighty lion roars.'
 7. 'Tired lions sleep.'
 8. 'Lion breathes peacefully.'
 9. 'Lion walks slowly.'
10. 'Lion cub grows strong.'

Top 10 prompts most aligned with NEGATIVE PC2 direction:
 1. 'That lion's mane color indicates nutritional status accurately.'
 2. 'That lion's mane color indicates nutritional status accurately.'
 3. 'That lion's hunting success correlates with environmental factors.'
 4. 'That lion's hunting success correlates with environmental factors.'
 5. 'That lion's territorial behavior varies with pride composition.'
 6. 'Lions demonstrate learning abilities through problem-solving tests.'
 7. 'Lions show varying learning rates for different survival skills.'
 8. 'The lion's reproductive success correlates with territory quality.'
 9. 'The lion's metabolism adjusts to food scarcity periods.'
10. 'That lion's territorial calls vary with audience composition.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC3 'lion' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7241 (raw projection: 9.6641)
Top positive #2: 0.6719 (raw projection: 8.4688)
Top positive #3: 0.6631 (raw projection: 7.7109)
Top negative #1: -0.5898 (raw projection: -8.3438)
Top negative #2: -0.5879 (raw projection: -8.0312)
Top negative #3: -0.5674 (raw projection: -7.0508)

Top 10 prompts most aligned with POSITIVE PC3 direction:
 1. 'Lions face eternal.'
 2. 'Lions show eternal.'
 3. 'Lions face all.'
 4. 'Lions adapt eternal.'
 5. 'Lions adapt infinity.'
 6. 'Lions show everything.'
 7. 'Wise lion decides.'
 8. 'Lions adapt existence.'
 9. 'Lion pride rests.'
10. 'Lion cub eternal.'

Top 10 prompts most aligned with NEGATIVE PC3 direction:
 1. 'Please describe the seasonal variations in lion behavior and activity patterns.'
 2. 'Please describe how climate change affects lion habitats and behavior.'
 3. 'Describe lion territories.'
 4. 'Please describe the life cycle of a lion from birth to death.'
 5. 'Please describe the role of play in lion cub development and learning.'
 6. 'Describe lion seasonal behavior.'
 7. 'Describe lion pack dynamics.'
 8. 'Describe lion wilderness life.'
 9. 'Please describe the hunting techniques lions use for different prey species.'
10. 'Describe lion family bonds.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC4 'lion' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.3867 (raw projection: 4.3359)
Top positive #2: 0.3450 (raw projection: 3.9219)
Top positive #3: 0.3450 (raw projection: 3.9219)
Top negative #1: -0.6226 (raw projection: -4.6875)
Top negative #2: -0.5957 (raw projection: -4.5898)
Top negative #3: -0.5825 (raw projection: -3.9512)

Top 10 prompts most aligned with POSITIVE PC4 direction:
 1. 'The vintage lion poster advertised a long-defunct traveling circus.'
 2. 'Lions exhibit complex grief responses to pride member deaths.'
 3. 'Lions exhibit complex grief responses to pride member deaths.'
 4. 'The brass lion fountain operated continuously for over fifty years.'
 5. 'The lion's thick mane protected him during fierce battles with other males.'
 6. 'Asian lions survive only in India's Gir Forest.'
 7. 'Lions exhibit sophisticated social structures rivaling those of primates.'
 8. 'The virtual lion existed solely within advanced computer simulation programs.'
 9. 'The holographic lion display attracted thousands of fascinated museum visitors.'
10. 'The lion's mane made him look even larger and more intimidating.'

Top 10 prompts most aligned with NEGATIVE PC4 direction:
 1. 'Lions show love.'
 2. 'Lions show hope.'
 3. 'Lions show respect.'
 4. 'Lions overcome hardship.'
 5. 'Lions show wisdom.'
 6. 'Lions show devotion.'
 7. 'Lions show loyalty.'
 8. 'Lions show loyalty.'
 9. 'Lions display honor.'
10. 'Lion cub wanders.'
================================================================================

================================================================================
--- ORTHOGONAL PERTURBATION ON CONCEPT: 'lion' ---
--- LAYER: 31 ---
================================================================================
Perturbing along first orthogonal direction (PC109)...

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
Original model completion: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.

--- Perturbing final token activation along orthogonal direction ---
Perturbation scale -100.0x:                                                                       
Perturbation scale -20.0x:   73  73  73  73   73   73   73   73   73   73   73   73   73   73   73   73   73   73  
Perturbation scale -10.0x: The lion is the second largest feline in the world and is known for its distinctive    414   414   414   414   414   414   414   414   414   414   414   414   414
Perturbation scale -5.0x: The lion is the second-largest feline in the world and is known for its distinctive golden-brown or tawny coat with a long mane in males that can range from 30 to 80 pounds in body weight and 4 to 6 feet in length from the head to the base of the tail.
Perturbation scale -2.5x: A lion is a large carnivorous mammal known for its distinctive golden-brown or tawny coat, shaggy mane in males, and powerful physique, typically found in savannas and grasslands of Africa and India.
Perturbation scale -1.5x: A lion is a large carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities in the savannas and grasslands of Africa and India.
Perturbation scale +0.0x: A lion is a large, carnivorous mammal with a distinctive golden-brown coat, a shaggy mane in males, and a powerful physique, known for its regal appearance and fierce hunting abilities.
Perturbation scale +1.5x: A lion is a large, carnivorous mammal with a distinctive mane, golden-brown coat, and a powerful build, typically characterized by its majestic mane, sharp claws, and a loud, roaring voice.
Perturbation scale +2.5x: A lion is a large, carnivorous mammal with a distinctive mane, golden-brown coat, and a powerful build, typically weighing between 265-550 pounds, and is known for its majestic roar.
Perturbation scale +5.0x: A lion, also known as a king of beasts, is a majestic, carnivorous mammal with a tawny or golden coat, a long mane, powerful jaws, sharp claws, and a distinctive roar.
Perturbation scale +10.0x: A lion, also known as Panthera leo, is a majestic, carnivorous mammal with a tawny, sandy, or golden coat, a tuftened tail, a powerful jaw, sharp claws, and imposing stature, typically weighing between700-1,700 pounds.
Perturbation scale +20.0x: Ạ̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣
Perturbation scale +100.0x: ̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣̣
================================================================================

================================================================================
--- ABLATION EXPERIMENT ON CONCEPT: 'lion' ---
--- LAYER: 31 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe a lion in one sentence.'
All PCs (centroid only): 
All PCs except largest (PC0 only): 
All PCs except largest two (PC0+PC1): 
Top 1 PCA ablated: 
Top 2 PCAs ablated: 
Top 3 PCAs ablated: 
================================================================================

################################################################################
### PLOTTING OVERALL RESULTS ###
################################################################################

