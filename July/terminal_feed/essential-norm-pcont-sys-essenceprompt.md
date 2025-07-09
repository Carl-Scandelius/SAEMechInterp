(interp) [cscandelius@holygpu8a19102 July]$ python run_analysis.py --script last_token --use_system_prompt
Running LastToken analysis...
USE_SYSTEM_PROMPT_FOR_MANIFOLD set to: True
PERTURB_ONCE set to: False

Configuration: PERTURB_ONCE is set to False

Configuration: USE_SYSTEM_PROMPT_FOR_MANIFOLD is set to True

Configuration: USE_NORMALIZED_PROJECTION is set to True

Loading checkpoint shards: 100%|███████████████████████████████████████| 4/4 [00:09<00:00,  2.44s/it]

################################################################################
### STARTING ANALYSIS FOR LAYER 0 ###
################################################################################

Extracting activations from layer 0...
Extracting activations: 100%|████████████████████████████████████| 1944/1944 [00:44<00:00, 44.14it/s]
Extracting activations from layer 0...
Extracting activations: 100%|██████████████████████████████████████| 247/247 [00:05<00:00, 44.45it/s]
Concept 'dog': Found 132 effective eigenvectors out of 1944 (threshold: 0.0000)
Concept 'lion': Found 33 effective eigenvectors out of 247 (threshold: 0.0000)

================================================================================
Running experiments for system prompt: 'You are a helpful assistant.'
User prompt: 'Please describe the essential features of a dog in one sentence.'
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC0 ---
Perturbation scale -100.0x: TheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheDogThe dog's descriptionassistant

A dog is a helpfulassistant

A dog is a helpfulassistant

A dog is a helpfulassistant

The dog is a helpfulassistant

A dog is a helpfulassistant

A dog is a helpfulassistant

A dog is a
Perturbation scale -20.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal characterized by its furry body, four legs, a tail, and a distinctive set of teeth and vocalizations, typically classified into various breeds based on size, coat type, and behavior.
Perturbation scale +20.0x: A dog is a domesticated mammal characterized by its furry body, four legs, a tail, a distinctive barking sound, and a highly social and loyal nature, often serving as a companion animal to humans.
Perturbation scale +100.0x: A dog is a domesticated mammal belonging to the family Canidae, characterized by essential features such as its physical attributes such as a body covered in fur, a tail, and a head,...features such as a head,a body covered in afeatures suchfeatures such the the thefeatures such as a head, a body covered in
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 1 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC1 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
Perturbation scale -20.0x: ://://://://://://://://://://://://://://:// the://:// the the ( theater.://assistant
Perturbation scale -10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of physical and behavioral traits, such as a wagging tail, barking vocalizations, and a strong instinct to protect and interact with its human family.
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall at the shoulder.
Perturbation scale +20.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale +100.0x: A dog is a domesticated, carnivorous mammal characterized by its upright posture, typically having four legs, a warm-blooded circulatory system, and a highly social animal that is capable of living in a pack, with a long tail, and a furry body, and a carnivorous head, with a short snout, and a long body
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 2 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC2 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
Perturbation scale -20.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly developed sense of smell, hearing, and social interaction, often serving as a loyal companion to humans.
Perturbation scale -10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically having a keen sense of smell and hearing, and often exhibiting loyalty and affection towards humans.
Perturbation scale +20.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically having a keen sense of smell and hearing, and often exhibiting loyalty and affection towards humans.
Perturbation scale +100.0x: A dog is a domesticated mammal belonging to the family Canidae, characterized by its furry coat, four legs, a tail, and a highly developed sense of smell, primarily carnivorous in diet, and often social and affectionate in nature.
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 3 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC3 ---
Perturbation scale -100.0x: A dog is a domesticated mammal typically characterized by its loyalty, playfulness, and ability to be trained for various tasks, with a broad range of breeds and sizes, often kept as pets for human companionship.
Perturbation scale -20.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically having a keen sense of smell and hearing, and often exhibiting loyalty and affection towards humans.
Perturbation scale -10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale +20.0x: A dog is a domesticated mammal characterized by its furry body, four legs, a tail, and a distinctive set of teeth and vocalizations, often exhibiting loyalty, intelligence, and a strong instinct to protect and interact with its human companions.
Perturbation scale +100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0, AXIS: 4 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC4 ---
Perturbation scale -100.0x: A dog is a domesticated mammal belonging to the family Canidae, characterized by a wagging tail, erect ears, and a varied range of sizes, shapes, and breeds, often serving as a loyal companion and working animal to humans.
Perturbation scale -20.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly developed sense of smell, hearing, and loyalty, often serving as a companion animal to humans.
Perturbation scale -10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly developed sense of smell, hearing, and loyalty, often serving as a companion animal to humans.
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +20.0x: A dog is a domesticated mammal characterized by its furry body, four legs, a tail, and a distinctive set of teeth and vocalizations, typically classified into various breeds based on size, coat type, and behavior.
Perturbation scale +100.0x: A dog is a domesticated mammal belonging to the family Canidae, characterized by its distinctive physical features, such as a furry body, a long snout, and four legs, as well as its unique behaviors, including loyalty, playfulness, and a strong instinct to protect its pack.
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC0 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.9014 (raw projection: 0.0681)
Top positive #2: 0.8911 (raw projection: 0.0844)
Top positive #3: 0.8809 (raw projection: 0.0821)
Top negative #1: -0.8574 (raw projection: -0.0859)
Top negative #2: -0.8477 (raw projection: -0.0767)
Top negative #3: -0.8257 (raw projection: -0.0779)

Top 10 prompts most aligned with POSITIVE PC0 direction:
 1. 'He has a deep, booming bark.'
 2. 'She loves to ride in the car, no matter the destination.'
 3. 'She loves the beach, chasing the waves back and forth.'
 4. 'The dog's life is a beautiful, albeit brief, journey.'
 5. 'The dog's joy is simple, pure, and infectious.'
 6. 'He loves to roll in the grass.'
 7. 'She is a natural at the agility course.'
 8. 'She loves to dig in the sand at the beach.'
 9. 'He loves to be the little spoon.'
10. 'He's afraid of the vacuum cleaner.'

Top 10 prompts most aligned with NEGATIVE PC0 direction:
 1. 'The celebrity's tiny teacup dog had its own social media account with millions of followers.'
 2. 'The first-ever domesticated dog is believed to have descended from an ancient wolf population.'
 3. 'Why does one dog in a neighborhood start a barking chain reaction with every other dog?'
 4. 'My neighbor's golden retriever dog is incredibly well-trained and never jumps on guests.'
 5. 'The new city ordinance requires every dog owner to carry proof of rabies vaccination.'
 6. 'That dog has a very endearing habit of resting his head on my knee while I work.'
 7. 'That dog has a bad habit of chewing on furniture when he gets bored or anxious.'
 8. 'The dog's incessant shedding means that dog hair is a permanent accessory on all my clothes.'
 9. 'The dog's joyous reunion with his soldier owner was a viral video that touched millions.'
10. 'That dog is a furry alarm clock with a wet nose and no snooze button.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC1 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7769 (raw projection: 0.0571)
Top positive #2: 0.7744 (raw projection: 0.0518)
Top positive #3: 0.7734 (raw projection: 0.0646)
Top negative #1: -0.7861 (raw projection: -0.0729)
Top negative #2: -0.7837 (raw projection: -0.0591)
Top negative #3: -0.7690 (raw projection: -0.0401)

Top 10 prompts most aligned with POSITIVE PC1 direction:
 1. 'Her playful nips are starting to get a little too hard.'
 2. 'He looks so proud when he carries a big stick.'
 3. 'Her boundless love is a gift I don't deserve but cherish.'
 4. 'Her playful nature is a daily reminder not to take life too seriously.'
 5. 'After a long day of hiking, the tired dog fell asleep before he even finished his dinner.'
 6. 'She makes even the worst days feel manageable.'
 7. 'Her endearing personality makes her impossible not to love.'
 8. 'She has taught me so much about patience and love.'
 9. 'What is the best way to keep my dog entertained while I'm away?'
10. 'The dog's memory is quite good; he always remembers the people who give him the best treats.'

Top 10 prompts most aligned with NEGATIVE PC1 direction:
 1. 'The Schnoodle is a Schnauzer-Poodle mix.'
 2. 'The Cocker Spaniel has long, floppy ears.'
 3. 'The dog's loyalty is an unbreakable bond.'
 4. 'The dog's friendship is a bond that transcends words.'
 5. 'The Labradoodle combines the traits of a Labrador and a Poodle.'
 6. 'The Pomapoo is a Pomeranian and Poodle mix.'
 7. 'The dog's loyalty is a rock in turbulent times.'
 8. 'The Mastador is a Mastiff and Labrador cross.'
 9. 'The dog's loyalty is a powerful force.'
10. 'A police dog is a valuable member of the force.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC2 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7188 (raw projection: 0.0543)
Top positive #2: 0.6934 (raw projection: 0.0447)
Top positive #3: 0.6758 (raw projection: 0.0427)
Top negative #1: -0.6997 (raw projection: -0.0566)
Top negative #2: -0.6689 (raw projection: -0.0390)
Top negative #3: -0.6553 (raw projection: -0.0521)

Top 10 prompts most aligned with POSITIVE PC2 direction:
 1. 'Come!'
 2. 'Do dogs dream?'
 3. 'How long do dogs live?'
 4. 'Sit!'
 5. 'Drop it!'
 6. 'Is he friendly with other dogs?'
 7. 'Off!'
 8. 'Fetch!'
 9. 'What a good boy!'
10. 'What do dogs dream about?'

Top 10 prompts most aligned with NEGATIVE PC2 direction:
 1. 'The dog, covered from nose to tail in mud, was not allowed back in the house until he had a thorough bath.'
 2. 'The dog was an accomplice in the toddler's escape, nudging the back door open for her.'
 3. 'The dog's ears perked up, and he let out a low growl, alerting us to someone approaching the door.'
 4. 'The little dog, with comical determination, tried to climb the stairs while carrying a ball too big for his mouth.'
 5. 'The dog's deep, rumbling growl was a clear warning to the approaching stranger.'
 6. 'Although he was a very small dog, his personality filled the entire house with joy and chaos.'
 7. 'That dog has a comical way of smiling, pulling back his lips to show his teeth.'
 8. 'The dog's soft, rhythmic breathing as he slept beside me was a comforting sound in the quiet house.'
 9. 'The dog's friendship is a steady, reliable flame in the flickering uncertainty of life.'
10. 'The dog's quiet, steadfast presence has been a comfort through many of life's ups and downs.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC3 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.8130 (raw projection: 0.0547)
Top positive #2: 0.8091 (raw projection: 0.0646)
Top positive #3: 0.7861 (raw projection: 0.0564)
Top negative #1: -0.6113 (raw projection: -0.0401)
Top negative #2: -0.5918 (raw projection: -0.0334)
Top negative #3: -0.5513 (raw projection: -0.0379)

Top 10 prompts most aligned with POSITIVE PC3 direction:
 1. 'If dogs could run a business, what would it be?'
 2. 'If a dog could have a job, what would it be?'
 3. 'If a dog could write a review of its owner, what would it say?'
 4. 'If a dog could send a text, what would it say?'
 5. 'If a dog could give advice to humans, what would it say?'
 6. 'If dogs could vote, what issues would matter to them?'
 7. 'If dogs could have a superpower, what would it be?'
 8. 'If dogs could paint, what would they create?'
 9. 'If dogs could host a TV show, what would it be about?'
10. 'If a dog could make a movie, what would the plot be?'

Top 10 prompts most aligned with NEGATIVE PC3 direction:
 1. 'The Alaskan Malamute was bred for hauling heavy freight.'
 2. 'The greyhound stretched languidly on the rug.'
 3. 'The Neapolitan Mastiff is known for its massive size and loose, wrinkled skin.'
 4. 'The Tibetan Spaniel served as a companion and watchdog in monasteries.'
 5. 'The American Eskimo Dog is a small, companionable Spitz-type breed.'
 6. 'His wet-dog smell filled the entire car.'
 7. 'The Irish Terrier is known for its fiery red coat and temperament.'
 8. 'The brave little dog stood his ground, barking fiercely at the much larger intruder.'
 9. 'The Pug's wrinkled face requires regular cleaning.'
10. 'The pet adoption fee covers spaying/neutering and initial vaccinations.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC4 'dog' direction (Layer 0) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7490 (raw projection: 0.0366)
Top positive #2: 0.7324 (raw projection: 0.0346)
Top positive #3: 0.7295 (raw projection: 0.0359)
Top negative #1: -0.5166 (raw projection: -0.0322)
Top negative #2: -0.4905 (raw projection: -0.0461)
Top negative #3: -0.4875 (raw projection: -0.0392)

Top 10 prompts most aligned with POSITIVE PC4 direction:
 1. 'He is a very magnificent and powerful animal.'
 2. 'He is a very athletic and powerful animal.'
 3. 'She is a very sharp and intelligent animal.'
 4. 'She is a very bright and perceptive animal.'
 5. 'He is a very striking and powerful-looking animal.'
 6. 'He is a very powerful and magnificent-looking animal.'
 7. 'He is a very strong and athletic animal.'
 8. 'He is a very striking and magnificent animal.'
 9. 'He is a very powerful and impressive animal.'
10. 'She is a very bright and intuitive animal.'

Top 10 prompts most aligned with NEGATIVE PC4 direction:
 1. 'She runs in her sleep, probably chasing dream squirrels.'
 2. 'During the thunderstorm, the terrified dog hid under the kitchen table, shivering uncontrollably.'
 3. 'Underneath the porch, the timid rescue dog watched the world with wary, uncertain eyes.'
 4. 'His tail started wagging the moment I picked up the leash.'
 5. 'Why do dogs wag their tails?'
 6. 'Could you grab the poop bags?'
 7. 'How do dogs feel about mail carriers?'
 8. 'How do dogs feel about rain?'
 9. 'When I came home, the guilty-looking dog had spread trash all over the kitchen floor.'
10. 'Let sleeping dogs lie.'
================================================================================

================================================================================
--- ORTHOGONAL PERTURBATION ON CONCEPT: 'dog' ---
--- LAYER: 0 ---
================================================================================
Perturbing along first orthogonal direction (PC132)...

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along orthogonal direction ---
Perturbation scale -100.0x: A dog is a domestic animal that belongs to the Canine mamm, A Dog, A Dog, A Dog, A Dog.
Perturbation scale -20.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically classified as a carnivore with a highly social and trainable nature.
Perturbation scale +20.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically classified as a carnivorous omnivore with a highly social and trainable nature.
Perturbation scale +100.0x: A dog, also known as a coll, is a canine, is a dog, is a canine, is a canine, is a dog, is a canine, is a canines, is a dog, is a), is a), is a dog, is a), is a dog, is a), is a dog, is a,
================================================================================

================================================================================
--- ABLATION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
All PCs (centroid only): 

 A

AA

 A A A A A A A A A A A A A A A A A A A A A A A A
All PCs except largest (PC0 only): 

 AングAA

 A A A A A A A A A A A A A A A A A A A A A A A A
All PCs except largest two (PC0+PC1): 

 A

AA

 A A A A A A A A A A A A A A A A A A A A A A A A
Top 1 PCA ablated: A dog is a domesticated mammal that is characterized by a carnivorous species, typically having a tail that is often trained to obey The essential features
Top 2 PCAs ablated: A dog is a domesticated mammal that is characterized by a carnivorous species, typically having a tail that is often trained to obey The essential features
Top 3 PCAs ablated: A dog is a domesticatedmammalI apologize I made a mistake. A dog is a domesticated mammal A dog is a mammal
================================================================================

################################################################################
### STARTING ANALYSIS FOR LAYER 15 ###
################################################################################

Extracting activations from layer 15...
Extracting activations: 100%|████████████████████████████████████| 1944/1944 [00:43<00:00, 44.41it/s]
Extracting activations from layer 15...
Extracting activations: 100%|██████████████████████████████████████| 247/247 [00:05<00:00, 44.71it/s]
Concept 'dog': Found 133 effective eigenvectors out of 1944 (threshold: 0.0056)
Concept 'lion': Found 28 effective eigenvectors out of 247 (threshold: 0.0331)

================================================================================
Running experiments for system prompt: 'You are a helpful assistant.'
User prompt: 'Please describe the essential features of a dog in one sentence.'
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC0 ---
Perturbation scale -100.0x:  answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered answered
Perturbation scale -20.0x:  dogs, a,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Perturbation scale -10.0x: A dog is a domestic animal, often a pet, with a variety of sizes, from small to large, from a variety of breeds, from a variety of different dogs, from a variety of different dogs, from a variety from a variety, from a variety, from a variety, from a dog, from a dog, from a dog, from
Perturbation scale -5.0x: Dogs are highly social, loyal, and adaptable animals, characterized by their unique characteristics, such as a wagging tail, floppy ears, a keen sense of smell, and a strong instinct to protect and care for their families.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by a furry body, a wagging tail, a keen sense of smell, and a loyal and social nature, often serving as a beloved companion and working animal.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, a keen sense of smell, and a loyal, social nature that makes them popular as companions and working animals.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its four legs, a furry coat, a wagging tail, and a distinctive barking or whining vocalization, typically belonging to the family Canidae.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its four legs, furry coat, and ability to bark, with essential features including a tail, ears, and a nose, typically belonging to the family Canidae.
Perturbation scale +5.0x: The essential features of a dog can be described as a mammalian species with the characteristics of four legs, a tail, and a single pair of limbs, belonging to the class Mammalia, order Carnivora, and having the ability to breathe, walk, and communicate through vocalizations.
Perturbation scale +10.0x: The essential features of a dog in one sentence

The essential features of a dog in one sentence

The essential features of a dog in one sentence

The essential features of a assistant in one sentence

 The essential features of a dog in oneassistant
Perturbation scale +20.0x:   
Perturbation scale +100.0x: swickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickswickórswickórswickórswickórswickórswickór
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 1 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC1 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://acellularaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganaganagan
Perturbation scale -20.0x: A dog can be described as a domestic animal that is characterized by its distinctive characteristics including a specific tipo de la lenga en una tradjala en una tradjajal en una tradjajal en una tradjajal en una tradjal en una tradjal en una tradjal en una tradjal en una trad
Perturbation scale -10.0x: A dog is a domesticated carnivorous mammal characterized by its distinctive physical characteristics, such as a furry coat, four legs, a tail, and a head with a snout, ears, and a brain, and behavioral traits including loyalty, adaptability, and a strong instinct for hunting and companionship.
Perturbation scale -5.0x: A dog is a domesticated carnivorous mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and claws, typically having a strong sense of smell, hearing, and loyalty to its human companions.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and claws, typically having a strong sense of smell, loyalty, and a wide range of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly developed sense of smell, hearing, and loyalty, often serving as a companion animal to humans.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a varied range of sizes, breeds, and temperaments, often serving as a loyal companion to humans.
Perturbation scale +5.0x: A dog is a domesticated mammal that belongs to the family Canidae, characterized by its furry coat, four legs, a wagging tail, and a range of vocalizations, with a highly variable size, shape, and behavior depending on its breed and individual characteristics.
Perturbation scale +10.0x: A dog is a domesticated mammal that belongs to the family Canidae, characterized by a wide range of physical and behavioral variations, but generally includes essential features such as a furry coat, a wagging tail, and a range of vocalizations to communicate.
Perturbation scale +20.0x: A dog is a mammor animal, typically a domestic animal, with a range of variations, ranging from 1 to 5, depending on the duration, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
Perturbation scale +100.0x: ://:// Angeles Angeles Angeles Angeles wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel wheel
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 2 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC2 ---
Perturbation scale -100.0x:  NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC NBC
Perturbation scale -20.0x: A dog is a fictional story by James Stevens, a 2005 novel by a man that is a fictional story about a man that is a fictional story.
Perturbation scale -10.0x: A dog is a domesticated mammal characterized by its unique physical features, such as a furry coat, a wagging tail, and a distinctive nose, and its endearing personality, which often includes loyalty, playfulness, and affectionate behavior.
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a distinctive barking or howling sound, often exhibiting loyalty, playfulness, and affection towards its human companions.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, often loyal, and trainable nature.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive barking or whining vocalization, typically serving as a loyal companion and often used for various tasks such as hunting, herding, or assistance.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive barking or whining vocalization, typically serving as a loyal companion and often used for various tasks such as hunting, herding, or assistance.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its upright posture, four legs, a wagging tail, and a varied range of sizes, breeds, and coat types, with a highly developed sense of smell and hearing, and the ability to form strong bonds with humans.
Perturbation scale +10.0x: A dog, also known as Canis lupus familiaris, is a domesticated mammal characterized by its quadrupedal stance, erect or slightly inclined ears, a varied range of coat types and colors, and a highly developed sense of hearing and smell, with the ability to communicate through vocalizations and body language.
Perturbation scale +20.0x: A canine, or "Dhriwta" in your native språket (Språket)  Sprog, Tyske, Språket, Språket, Språket, Språket, Språket, Språet, Språet, Språet, Språet, Språet,
Perturbation scale +100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 3 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC3 ---
Perturbation scale -100.0x: ezezezezezezezezezezezezezseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseauseau
Perturbation scale -20.0x: A dog is a mammalian animal that consists of several key features, including a canine anatomy, a unique life cycle, and a range of essential characteristics, including its distinctive skeletal structure, a specific set of senses, and a set of essential features that can be tailored to improve its survival and adaptability. 

Here are the key features of a dog
Perturbation scale -10.0x: A dog is a domesticated mammal characterized by its unique combination of physical features, including a furry coat, four legs, a distinctive skeletal structure, and a highly developed brain, as well as essential behavioral traits such as loyalty, social interaction, and a strong prey drive.
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of physical and behavioral features, including a keen sense of smell, loyalty, and a wide range of breeds that vary in size, shape, and temperament.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of physical and behavioral features, including a keen sense of smell, loyalty, and a wide range of breeds and sizes.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of physical and behavioral features, including a keen sense of smell, loyalty, and a wide range of breeds and sizes.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, often loyal, and trainable nature.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, often loyal, and trainable nature.
Perturbation scale +5.0x: A dog is a domesticated, carnivorous mammal characterized by a furry body, a wagging tail, four legs, and a distinctive barking or whining vocalization, often used as a companion animal or working animal.
Perturbation scale +10.0x: A dog is a domesticated, carnivorous mammal characterized by a furry body, a wagging tail, and a barking or whining vocalization, often used as a companion or working animal.
Perturbation scale +20.0x: A dog is typically a small to large animal, often referring to something, referring to something, referring to something, referring to something, referring to something, referring to something, referring to something, referring to something, referring to something, referring to something, referring to something, referring to something, referring to something, referring to something, referring to something
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
.
.
.
.
.
.
.
ingroup.
.
.
.
ingroup.
ingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroupingroup
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15, AXIS: 4 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC4 ---
Perturbation scale -100.0x: ://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://://
Perturbation scale -20.0x: A dog is a domesticated mammal that can be described as a quadrilateral mammal with a variable number of limbs, a unique ability to bark and produce various sounds, and the ability to be of any size, shape, and color, with the ability to be trained and domesticated.
Perturbation scale -10.0x: A dog is a domesticated mammal characterized by its ability to be trained, its varied sizes and breeds, a wagging tail, a barking or howling sound, and a unique ability to form strong emotional bonds with humans.
Perturbation scale -5.0x: A dog is a domesticated, carnivorous mammal characterized by a furry body, four legs, a tail, and a distinctive barking or howling sound, typically serving as a loyal companion to humans.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry body, four legs, a tail, and a distinctive barking or whining vocalization, typically serving as a loyal companion and often used for various tasks such as hunting, herding, or assistance.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry body, four legs, a tail, and a distinctive set of teeth and vocalizations, typically serving as a loyal companion and often used for various tasks such as hunting, herding, and assistance.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically classified as a carnivore with a highly social and trainable nature.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a distinctive barking or whining vocalization, often exhibiting loyalty, affection, and a strong instinct to protect and please its human companions.
Perturbation scale +10.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a range of behaviors that include loyalty, affection, and a strong instinct to communicate and interact with its human family and environment.
Perturbation scale +20.0x: A dog is a domesticated animal that is characterized by a range of essential features, including a furry coat, a distinctive body shape, a loud and varied vocalization, a strong and energetic behavior, a ability to train and learn, and a affectionate and empathetic behavior towards humans.
Perturbation scale +100.0x: ‌imanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimanimaniman
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC0 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.6357 (raw projection: 1.1973)
Top positive #2: 0.6172 (raw projection: 1.1846)
Top positive #3: 0.6094 (raw projection: 1.1992)
Top negative #1: -0.9106 (raw projection: -5.7461)
Top negative #2: -0.9106 (raw projection: -5.5273)
Top negative #3: -0.9033 (raw projection: -5.6328)

Top 10 prompts most aligned with POSITIVE PC0 direction:
 1. 'The dog's devotion to its family is a powerful bond.'
 2. 'The dog's loyalty is a comforting and steady presence.'
 3. 'The dog's joy is simple, pure, and infectious.'
 4. 'The sheepdog herded the flock with impressive skill.'
 5. 'The dog's companionship is a constant source of joy.'
 6. 'The dog's powerful legs propelled him through the snow with surprising ease.'
 7. 'The dog's loyalty is a constant source of comfort.'
 8. 'His soft snoring is a comforting sound at night.'
 9. 'The dog's loyalty is a powerful and humbling thing.'
10. 'The dog's friendship is a constant source of comfort and joy.'

Top 10 prompts most aligned with NEGATIVE PC0 direction:
 1. 'How do I help my dog cope with thunderstorms?'
 2. 'How can I tell if my dog is happy?'
 3. 'What are the best ways to exercise my dog indoors?'
 4. 'What is the best way to clean my dog's teeth?'
 5. 'What are the best ways to bond with my dog?'
 6. 'What is the best way to teach my dog to swim?'
 7. 'What are the best treats for training dogs?'
 8. 'What are the best dog breeds for therapy work?'
 9. 'What is the best way to travel with a dog?'
10. 'How do I help my dog with separation anxiety?'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC1 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7095 (raw projection: 2.0469)
Top positive #2: 0.6660 (raw projection: 1.6338)
Top positive #3: 0.6636 (raw projection: 1.7881)
Top negative #1: -0.6743 (raw projection: -2.2031)
Top negative #2: -0.6733 (raw projection: -1.8691)
Top negative #3: -0.6602 (raw projection: -1.8662)

Top 10 prompts most aligned with POSITIVE PC1 direction:
 1. 'He has a heart of gold.'
 2. 'He is a very gentle giant.'
 3. 'He is a very good listener.'
 4. 'He is a very true and steadfast friend.'
 5. 'Her fur is as soft as silk.'
 6. 'The dog's heart is full of love.'
 7. 'He loves to roll in the grass.'
 8. 'His breath is terrible.'
 9. 'He is a very steadfast and true friend.'
10. 'He chased his tail in circles.'

Top 10 prompts most aligned with NEGATIVE PC1 direction:
 1. 'My dog has a very specific, and very loud, bark reserved for the squirrel that taunts him daily.'
 2. 'My dog's worst fear is the sound of the nail clippers, which sends him scurrying under the bed.'
 3. 'The dog's comical sneeze, a full-body event, happened every time he sniffed a dandelion.'
 4. 'The dog's nose, a marvel of biological engineering, contains up to 300 million olfactory receptors.'
 5. 'The dog's DNA test revealed a surprising mix of five different breeds, including Chihuahua and Great Dane.'
 6. 'My dog has a very specific routine that involves napping in three different sunny spots throughout the day.'
 7. 'The ancient Roman mosaics often depicted a 'beware of the dog' warning at the entrance of homes.'
 8. 'My dog's greeting is a full-body experience, starting with a tail wag and ending in a flurry of kisses.'
 9. 'That dog has a very specific, high-pitched bark he reserves only for the mail carrier.'
10. 'The dog was found, thanks to his microchip, three towns away from where he went missing.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC2 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.6265 (raw projection: 1.5986)
Top positive #2: 0.6216 (raw projection: 1.5996)
Top positive #3: 0.6206 (raw projection: 1.5049)
Top negative #1: -0.5420 (raw projection: -1.9004)
Top negative #2: -0.5259 (raw projection: -1.3965)
Top negative #3: -0.4934 (raw projection: -1.5225)

Top 10 prompts most aligned with POSITIVE PC2 direction:
 1. 'The Belgian Malinois is often used by military and police forces.'
 2. 'The Australian Shepherd is known for its high energy levels.'
 3. 'The Wire Fox Terrier is a lively and intelligent breed.'
 4. 'The Kuvasz is a large Hungarian guardian breed.'
 5. 'The German Wirehaired Pointer is a versatile hunting dog.'
 6. 'The Scottish Deerhound is one of the tallest dog breeds.'
 7. 'The Xoloitzcuintli is a hairless breed from Mexico.'
 8. 'The Akita is a symbol of loyalty in Japan.'
 9. 'The Black and Tan Coonhound is an American scent hound.'
10. 'The Iditarod is a famous sled dog race in Alaska.'

Top 10 prompts most aligned with NEGATIVE PC2 direction:
 1. 'That dog has a face that could launch a thousand belly rubs, and he knows how to use it.'
 2. 'The dog's unwavering stare could bore holes through steel, especially when pizza was present.'
 3. 'That dog is a living, breathing heating pad on cold nights, and he asks for nothing in return.'
 4. 'My dog, a master of stealth, can make a whole sandwich disappear from the counter without a sound.'
 5. 'With a look of sheer betrayal, the dog watched me eat the last bite of the bacon.'
 6. 'That dog is a professional crumb-catcher, strategically positioned under the baby's high chair.'
 7. 'The dog's mournful eyes looked up at me, begging for just one more bite of my dinner.'
 8. 'She runs in her sleep, probably chasing dream squirrels.'
 9. 'My dog considers it his solemn duty to protect the house from the menace of falling leaves.'
10. 'That dog has a deep-seated distrust of men wearing hats, a mystery we've never solved.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC3 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.6084 (raw projection: 1.7295)
Top positive #2: 0.6035 (raw projection: 1.6748)
Top positive #3: 0.6030 (raw projection: 2.0273)
Top negative #1: -0.6064 (raw projection: -1.4004)
Top negative #2: -0.5791 (raw projection: -1.2627)
Top negative #3: -0.5781 (raw projection: -1.3467)

Top 10 prompts most aligned with POSITIVE PC3 direction:
 1. 'He is the reason we have to vacuum every day.'
 2. 'He's still learning not to jump on people.'
 3. 'He's the reason my camera roll is always full.'
 4. 'She is the unofficial neighborhood watch.'
 5. 'He can sense when I'm feeling sad.'
 6. 'He has a bad habit of jumping on guests.'
 7. 'He's snoring louder than my uncle.'
 8. 'I need to trim his nails.'
 9. 'He's a professional food thief.'
10. 'He knows the difference between 'walk' and 'work'.'

Top 10 prompts most aligned with NEGATIVE PC3 direction:
 1. 'The dog's joy was palpable as he ran freely on the sandy beach for the very first time.'
 2. 'The dog's bond with the family's cat was an unusual but beautiful friendship to behold.'
 3. 'The dog sat by the window for hours, patiently waiting for the children to come home from school.'
 4. 'The dog, a blur of fur and motion, joyfully chased the ball down the long hallway.'
 5. 'The dog's gentle and patient nature with the new litter of kittens was a beautiful sight.'
 6. 'The sled dog team worked in perfect harmony, pulling the heavy sled across the frozen tundra.'
 7. 'The dog's soft, rhythmic breathing as he slept beside me was a comforting sound in the quiet house.'
 8. 'The dog's playful wrestling with his best friend, a Golden Retriever, was a joy to watch.'
 9. 'The dog's low, rumbling snore is a constant, comforting presence in the quiet house.'
10. 'The farm dog worked tirelessly from dawn until dusk, a loyal and indispensable helper.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC4 'dog' direction (Layer 15) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.4236 (raw projection: 1.1914)
Top positive #2: 0.4143 (raw projection: 0.9766)
Top positive #3: 0.3911 (raw projection: 1.2695)
Top negative #1: -0.4797 (raw projection: -1.3604)
Top negative #2: -0.4333 (raw projection: -1.2266)
Top negative #3: -0.4214 (raw projection: -1.1797)

Top 10 prompts most aligned with POSITIVE PC4 direction:
 1. 'He looks so pathetic when he's wet.'
 2. 'He's very gentle when taking a treat from my hand.'
 3. 'He's a bit of a drama queen when his dinner is a minute late.'
 4. 'She has a very gentle mouth when taking treats.'
 5. 'Her playful nips are starting to get a little too hard.'
 6. 'He is a bit of a hoarder when it comes to his toys.'
 7. 'He's in the doghouse for chewing my favorite shoes.'
 8. 'Her endearing antics are the best part of coming home.'
 9. 'He gave me a look of utter betrayal when I left for work.'
10. 'He is very particular about where he does his business.'

Top 10 prompts most aligned with NEGATIVE PC4 direction:
 1. 'The Iditarod is a famous sled dog race in Alaska.'
 2. 'The Field Spaniel is a medium-sized breed in the spaniel family.'
 3. 'The Lakeland Terrier is a small, sturdy dog from the Lake District of England.'
 4. 'The Black and Tan Coonhound is an American scent hound.'
 5. 'The Tibetan Mastiff is an ancient breed known for its protective instincts.'
 6. 'The Kuvasz is a large Hungarian guardian breed.'
 7. 'Describe a dog's favorite place in the world.'
 8. 'The Xoloitzcuintli is a hairless breed from Mexico.'
 9. 'The Morkie is a cross between a Maltese and a Yorkshire Terrier.'
10. 'The Cairn Terrier is a small, plucky breed from Scotland.'
================================================================================

================================================================================
--- ORTHOGONAL PERTURBATION ON CONCEPT: 'dog' ---
--- LAYER: 15 ---
================================================================================
Perturbing along first orthogonal direction (PC133)...

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along orthogonal direction ---
Perturbation scale -100.0x: chnerchnerwlchnerchnerchnerchnerchnerchnerwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwlwl
Perturbation scale -20.0x: Essertain though aparturenurenurenurenayersurenayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayersayers
Perturbation scale -10.0x: The essential features of a dog are the following: can be found with 1: certain key features are 2: however there are 2: however there are 2: however there are 2: 2: 2: 2: 2: 2: 2: 2: 2: 2: 
Perturbation scale -5.0x: The essential features of a dog include its physical characteristics such as a furry body, a distinctive nose, and a unique set of physical features that vary depending on the breed, as well as its behavioral traits such as loyalty, playfulness, and a strong instinct to follow basic commands.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its distinctive physical features, such as a furry coat, a wagging tail, and a generally quadrupedal stance, along with its unique behaviors and abilities, including the ability to learn and respond to commands and exhibit strong loyalty and affection towards its human companions.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry body, four legs, a tail, and a distinctive set of physical and behavioral features that vary depending on its breed, including a keen sense of smell, loyalty, and a wide range of vocalizations.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by a furry body, four legs, a tail, a muzzle, and a brain capable of complex behaviors, often serving as a loyal companion to humans.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by a furry body, four legs, a wagging tail, a keen sense of smell, and a highly social and loyal nature.
Perturbation scale +5.0x: A dog is a domesticated mammal with a wagging tail, typically furry, four-legged, and often with a friendly expression, that is usually between 20-70 cm in length, weighing between 5-80 kg, and is known for its loyalty and affection towards humans.
Perturbation scale +10.0x: A dog is a domestic animal, typically with a mammalian appearance, with a wide range of colors and a friendly spirit, usually with a body, a brain, a mind, and a friend, where you can, you, you, you, you, you, you, you, you, you, you, you, you, you,
Perturbation scale +20.0x:  A dog, a world, a ESP, A, A, A, A, A, A, A, A, A, A, A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
Perturbation scale +100.0x:  Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow Blow
================================================================================

================================================================================
--- ABLATION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 15 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
All PCs (centroid only): 
All PCs except largest (PC0 only): 
All PCs except largest two (PC0+PC1): 
Top 1 PCA ablated: The
Top 2 PCAs ablated: The
Top 3 PCAs ablated: The
================================================================================

################################################################################
### STARTING ANALYSIS FOR LAYER 31 ###
################################################################################

Extracting activations from layer 31...
Extracting activations: 100%|████████████████████████████████████| 1944/1944 [00:43<00:00, 44.80it/s]
Extracting activations from layer 31...
Extracting activations: 100%|██████████████████████████████████████| 247/247 [00:05<00:00, 44.95it/s]
Concept 'dog': Found 143 effective eigenvectors out of 1944 (threshold: 0.0794)
Concept 'lion': Found 30 effective eigenvectors out of 247 (threshold: 0.4233)

================================================================================
Running experiments for system prompt: 'You are a helpful assistant.'
User prompt: 'Please describe the essential features of a dog in one sentence.'
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 0 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC0 ---
Perturbation scale -100.0x: neyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyneyney
Perturbation scale -20.0x: Dogs, classified as the familyswaparnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarnaarna
Perturbation scale -10.0x: Dogs, classified as the family Canidae, have essential features that include their ability to stand upright, have retractable, non-retraction, or permanent, non-retractable, retractable, and permanent, retractable, and permanent, retractable, and permanent, retractable, and permanent, retractable, and permanent, retractable,
Perturbation scale -5.0x: Dogs are domesticated, carnivorous mammals belonging to the family Canidae, characterized by their upright, wagging tail, floppy ears, and varied coat types, which come in a wide range of shapes, sizes, and colors.
Perturbation scale -2.5x: A dog is a domesticated, carnivorous mammal characterized by its upright posture, four legs, a wagging tail, and a varied range of sizes, shapes, and coat types, with the ability to communicate through vocalizations and body language.
Perturbation scale -1.5x: A dog is a domesticated, carnivorous mammal characterized by its upright posture, four legs, a wagging tail, and a varied range of sizes, shapes, and coat types, with a highly developed sense of smell and hearing, and the ability to form strong emotional bonds with humans.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by a furry body, four legs, a tail, a distinctive barking or whining vocalization, and a highly social and trainable nature.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by a furry body, four legs, a tail, a distinctive barking or whining vocalization, and a highly social and trainable nature.
Perturbation scale +5.0x: A dog typically has a furry body, four legs, a tail, ears, a snout or muzzle, and a distinctive barking or vocalization system, often serving as a companion animal to humans.
Perturbation scale +10.0x: A dog typically includes characteristics such as a furry or scaled body fur (can sometimes be bald), four legs (although some breeds may be three-legged due to injury or surgery), a tail (usually long or short), ears (usually triangular or rounded tips), a snout (usually triangular tip), a canine dental arrangement (including I1 I1 I
Perturbation scale +20.0x: assistantANJIassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistant
Perturbation scale +100.0x: assistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistant
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 1 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC1 ---
Perturbation scale -100.0x: translatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslatedtranslated
Perturbation scale -20.0x: Essentially a dog is a domesticated carnivorous mammal typically translated into English translation translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated translated
Perturbation scale -10.0x: A dog is a domesticated carnivorous mammal characterized by a furry body, a wet nose, a pair of ears, a tail, four legs, and a brain capable of complex behavior and trainability, typically living in close companionship with humans.
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, a distinctive bark, and a highly social and trainable nature, typically classified into various breeds based on size, shape, and behavior.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, a distinctive barking sound, and a highly social and trainable nature, often serving as a loyal companion to humans.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, a distinctive barking sound, and a highly social and loyal nature, often serving as a companion animal to humans.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly social, loyal, and trainable nature, often serving as a companion animal to humans.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry body, four legs, a tail, and a distinctive barking or whining vocalization, often serving as a loyal companion and domesticated pet.
Perturbation scale +5.0x: A dog is a domesticated, carnivorous mammal characterized by its upright posture, four legs, a wagging tail, and a variety of breeds with unique physical and behavioral traits.
Perturbation scale +10.0x: A dog is a domesticated, carnivorous mammal characterized by its unique combination of physical features, such as a wagging tail, floppy or erect ears, and a varied range of sizes, shapes, and coat types.
Perturbation scale +20.0x: A dog is a domesticated, often sociable, mammal with characteristics such as fur or hair, claws, and sharp or blunt canines, with species varying greatly in size, shape, and behavioral traits.
Perturbation scale +100.0x: olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1olen1
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 2 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC2 ---
Perturbation scale -100.0x:   (  ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( (
Perturbation scale -20.0x: Dogs (Canis lupus (or (C. ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( (
Perturbation scale -10.0x: A dog is a domesticated, often carnivorous, mammal (Canis lupus familiaris) that is a member of the Hound family (Canidae), typically characterized by a four-legged, often furry, and usually endearing or energetic body.
Perturbation scale -5.0x: A dog is a domesticated, carnivorous mammal that is a member of the family Canidae, characterized by a varied range of sizes, coat types, and colors, a wagging tail, and a highly social and often loyal nature.
Perturbation scale -2.5x: A dog is a domesticated, carnivorous mammal characterized by a furry body, a wagging tail, four legs, a snout or muzzle, and a highly developed sense of hearing, smell, and social interaction.
Perturbation scale -1.5x: A dog is a domesticated, carnivorous mammal characterized by a furry body, a wagging tail, four legs, a snout, and a highly developed sense of smell and hearing, often serving as a loyal companion to humans.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically classified as a carnivore with a highly social and trainable nature.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive nose, ears, and mouth, typically weighing between 5-200 pounds and living an average lifespan of 10-17 years.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, tail, ears, and nose, typically having a carnivorous diet, social behavior, and ability to communicate through vocalizations and body language.
Perturbation scale +10.0x: A dog typically possesses characteristics such as four legs, furry coat, tail, nose, ears, teeth, claws, ability to bark, loyalty towards owners, intelligence, agility, and various breeds-specific traits.
Perturbation scale +20.0x: A dog possesses essential features consisting Animalia kingdom membership Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia Animalia
Perturbation scale +100.0x: elseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelseaelsea
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 3 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along PC3 ---
Perturbation scale -100.0x: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Perturbation scale -20.0x: A dog, a domesticated, Carnivorous, Canid, is characterized by its distinctive features, including a furry, four-legged, and usually, a barking, social, and highly trainable, semi-intelligent, and endearing, homely, and a wide, varied, and, to a human, a, to a human,
Perturbation scale -10.0x: A dog is a domesticated, carnivorous, and social mammal characterized by a furry, four-legged body, a wagging tail, a distinctive barking or whining voice, and a highly variable, but generally loyal, friendly, and trainable, temperament.
Perturbation scale -5.0x: A dog is a domesticated, carnivorous mammal characterized by a furry, four-legged body, a wagging tail, and a highly social, trainable, and affectionate nature.
Perturbation scale -2.5x: A dog is a domesticated, carnivorous mammal characterized by a furry body, a wagging tail, four legs, a distinctive barking or whining vocalization, and a highly social, loyal, and trainable nature.
Perturbation scale -1.5x: A dog is a domesticated, carnivorous mammal characterized by a furry body, four legs, a tail, and a distinctive set of teeth and claws, often exhibiting loyalty, intelligence, and a wide range of behaviors and emotions.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically classified as a carnivore with a highly social and trainable nature.
Perturbation scale +5.0x: A dog is a domesticated mammal characterized by its upright posture, four legs, a wagging tail, floppy ears, and a variety of sizes and breeds that often exhibit loyalty, intelligence, and affectionate behavior towards humans.
Perturbation scale +10.0x: The essential features of a dog typically include a mammalian body with four legs, a tail, and often a furry coat, along with distinct facial features such as ears and a nose, and the ability to communicate through various vocalizations and body language.
Perturbation scale +20.0x: The essential features of a dog typically include a mammalian body covered in fur or hair, typically four legs and often a tail attached near the rear of their body which assists them with balance and communication among other aspects such as ears which assist them with hearing and assisting them navigate their environment among many other vital aspects which assist them navigate their environment among many other
Perturbation scale +100.0x: /assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets/assets
================================================================================

================================================================================
--- INTERVENTION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31, AXIS: 4 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

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

Perturbation scale -20.0x:  Dogs typically consistWidgetItemassistant

 Dogs typically consistWidgetItem
Perturbation scale -10.0x: A dog typically possesses characteristics such as canine physiology, quadruped locomotion, keen senses including smell and hearing, loyalty towards humans, varied coat textures and colors, intelligence trainable behavior, pack mentality instincts, vocal communication through bark sounds, sharp canine teeth, claws for grasping surfaces, agile limbs enabling agility movements, strong canine jaws enabling powerful bites strength
Perturbation scale -5.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, sharp canine teeth, keen sense of smell, floppy ears, wagging tail, and ability to communicate through vocalizations and body language.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, sharp teeth, and a highly developed sense of smell, hearing, and loyalty towards its human companions.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a wagging tail, and a highly developed sense of smell, hearing, and loyalty, often serving as a companion animal to humans.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated, carnivorous mammal characterized by a furry body, a wagging tail, four legs, and a distinctive barking or whining vocalization, often serving as a loyal companion to humans.
Perturbation scale +2.5x: A dog is a domesticated, carnivorous mammal characterized by a furry body, a wagging tail, four legs, and a distinctive barking or howling vocalization, often serving as a loyal companion to humans.
Perturbation scale +5.0x: A dog is a domesticated, carnivorous mammal that is characterized by a furry, four-legged body, a wagging tail, and a range of physical and behavioral traits that can vary greatly depending on the specific breed, but generally includes a strong sense of loyalty, a keen sense of smell, and a capacity for learning and social interaction.
Perturbation scale +10.0x: A dog is a domestic, (carnivorous)  (mammal)  (vertebrate)  (mammal)  (mammal)  (mammal)  (mammal)  (mammal)  (mammal)  (mammal)  (mammal
Perturbation scale +20.0x: A '' ('' ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( (
Perturbation scale +100.0x:  ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( (
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC0 'dog' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.8457 (raw projection: 24.2656)
Top positive #2: 0.8389 (raw projection: 26.0938)
Top positive #3: 0.8350 (raw projection: 23.3594)
Top negative #1: -0.6885 (raw projection: -4.8516)
Top negative #2: -0.6768 (raw projection: -4.2070)
Top negative #3: -0.6641 (raw projection: -4.7852)

Top 10 prompts most aligned with POSITIVE PC0 direction:
 1. 'What is the best way to teach my dog to swim?'
 2. 'How do I teach my dog to come when called?'
 3. 'What is the best way to teach my dog to back up on command?'
 4. 'How can I tell if my dog is happy?'
 5. 'How do I help my dog with car sickness?'
 6. 'How do I stop my dog from chewing on furniture?'
 7. 'How do I help my dog overcome fear of loud noises?'
 8. 'How do I stop my dog from barking excessively?'
 9. 'What is the best way to teach my dog to crawl?'
10. 'How do I stop my dog from digging in the yard?'

Top 10 prompts most aligned with NEGATIVE PC0 direction:
 1. 'The dog's quiet understanding and lack of judgment make him the perfect confidant.'
 2. 'The dog's devotion to its family is a powerful bond.'
 3. 'The dog's friendship is a constant source of comfort and joy.'
 4. 'The dog's companionship is a constant source of joy.'
 5. 'The farm dog worked tirelessly from dawn until dusk, a loyal and indispensable helper.'
 6. 'The dog's capacity for happiness is truly limitless.'
 7. 'The dog's keen senses make him an excellent watchdog, alerting us to any unusual activity.'
 8. 'The dog's companionship has brought immeasurable happiness to my life.'
 9. 'The dog's powerful legs propelled him through the snow with surprising ease.'
10. 'The dog's gentle presence has a profoundly calming effect on the students in the special education classroom.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC1 'dog' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.7456 (raw projection: 12.9062)
Top positive #2: 0.7373 (raw projection: 13.0859)
Top positive #3: 0.7358 (raw projection: 12.4766)
Top negative #1: -0.6367 (raw projection: -8.6250)
Top negative #2: -0.6289 (raw projection: -6.7852)
Top negative #3: -0.6157 (raw projection: -6.6211)

Top 10 prompts most aligned with POSITIVE PC1 direction:
 1. 'If a dog could give advice to humans, what would it say?'
 2. 'If a dog could write a thank-you note to its owner, what would it say?'
 3. 'If a dog could write a thank-you note, what would it say?'
 4. 'If a dog could write a review of its owner, what would it say?'
 5. 'What is the most important rule in a dog's world?'
 6. 'If a dog could invent a gadget, what would it be?'
 7. 'If a dog could be any other animal for a day, what would it choose?'
 8. 'What is the bravest thing a dog has ever done?'
 9. 'If dogs could write a letter to their younger selves, what would they say?'
10. 'If dogs could vote, what issues would matter to them?'

Top 10 prompts most aligned with NEGATIVE PC1 direction:
 1. 'That dog is surprisingly delicate when taking a treat from my hand, using only his lips.'
 2. 'The Dalmatian's spots are unique to each individual.'
 3. 'The Shar-Pei is famous for its deep wrinkles.'
 4. 'My dog has a very specific routine that involves napping in three different sunny spots throughout the day.'
 5. 'My dog loves to give hugs, which involves him placing his big paws on my shoulders.'
 6. 'The veterinarian was very patient with my nervous dog.'
 7. 'The dog show features an agility competition.'
 8. 'The dog's diet consists of a special kibble formulated for canines with sensitive stomachs.'
 9. 'I bought a new orthopedic bed to help my aging dog with his joint pain.'
10. 'The rescue dog was initially very timid, but now he demands belly rubs from everyone he meets.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC2 'dog' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.5137 (raw projection: 5.0039)
Top positive #2: 0.4995 (raw projection: 4.3203)
Top positive #3: 0.4958 (raw projection: 4.2812)
Top negative #1: -0.6050 (raw projection: -6.4648)
Top negative #2: -0.5903 (raw projection: -6.9766)
Top negative #3: -0.5850 (raw projection: -7.7539)

Top 10 prompts most aligned with POSITIVE PC2 direction:
 1. 'The dog's contribution to the search and rescue effort was honored with a special medal.'
 2. 'The dog's bond with the family's cat was an unusual but beautiful friendship to behold.'
 3. 'She is a very lively and high-spirited creature.'
 4. 'She waits by the window for the kids to come home from school.'
 5. 'The dog's gentle and patient nature with the new litter of kittens was a beautiful sight.'
 6. 'He is a very noble and dignified-looking creature.'
 7. 'The dog park was full of energetic pups this morning.'
 8. 'The dog's frantic barking alerted the family to the fire, saving all of their lives.'
 9. 'He is a very majestic and noble-looking creature.'
10. 'She curled up at the foot of the bed and fell fast asleep.'

Top 10 prompts most aligned with NEGATIVE PC2 direction:
 1. 'He's a very patient dog who will wait forever for a dropped crumb.'
 2. 'He seems to think he's a lap dog, despite weighing 80 pounds.'
 3. 'He has a talent for finding the muddiest puddle.'
 4. 'The dog's incessant shedding means that dog hair is a permanent accessory on all my clothes.'
 5. 'He's snoring louder than my uncle.'
 6. 'He's not a barker, but he does a lot of low 'woofing'.'
 7. 'He's the reason my camera roll is always full.'
 8. 'He's a terrible guard dog; he'd just lick an intruder.'
 9. 'That dog has more energy than a toddler on sugar.'
10. 'That dog is a bed hog, somehow managing to take up the entire king-sized mattress.'
================================================================================

================================================================================
--- Analyzing original dataset prompts along the PC3 'dog' direction (Layer 31) ---
Using normalized projections (projection magnitude / vector magnitude)

Normalized projection values:
Top positive #1: 0.4976 (raw projection: 9.2109)
Top positive #2: 0.4958 (raw projection: 9.3672)
Top positive #3: 0.4924 (raw projection: 9.0156)
Top negative #1: -0.4495 (raw projection: -14.3984)
Top negative #2: -0.4475 (raw projection: -14.1094)
Top negative #3: -0.4382 (raw projection: -18.5312)

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
Top positive #1: 0.6050 (raw projection: 6.5625)
Top positive #2: 0.5664 (raw projection: 5.7188)
Top positive #3: 0.5244 (raw projection: 7.1914)
Top negative #1: -0.4680 (raw projection: -4.6445)
Top negative #2: -0.4675 (raw projection: -3.8359)
Top negative #3: -0.4546 (raw projection: -10.3984)

Top 10 prompts most aligned with POSITIVE PC4 direction:
 1. 'I can't stay mad at that face for long.'
 2. 'I wish I had his carefree attitude.'
 3. 'I think I'm more excited for his birthday than he is.'
 4. 'He's a rescue, but sometimes I think he rescued me.'
 5. 'He has a flair for the dramatic, especially when he wants something.'
 6. 'He's not allowed on the couch, but he sneaks up when we're not looking.'
 7. 'He's my shadow, following me from one room to the next.'
 8. 'He's a little bit of a diva about his sleeping arrangements.'
 9. 'His tail could clear a coffee table in a single wag.'
10. 'He's getting a bit gray around the muzzle.'

Top 10 prompts most aligned with NEGATIVE PC4 direction:
 1. 'The Skye Terrier has long hair that parts down its back.'
 2. 'The dog shelter is always in need of blankets and food.'
 3. 'Describe a dog's reaction to seeing snow for the first time.'
 4. 'The Redbone Coonhound is known for its beautiful solid red coat.'
 5. 'The Treeing Walker Coonhound is a fast and able hunter.'
 6. 'The Basset Hound's long ears help scoop scents towards its nose.'
 7. 'The Pomapoo is a Pomeranian and Poodle mix.'
 8. 'The Ibizan Hound is an elegant and agile sighthound.'
 9. 'Write a letter from a dog to its owner.'
10. 'The Weimardoodle is a Weimaraner and Poodle mix.'
================================================================================

================================================================================
--- ORTHOGONAL PERTURBATION ON CONCEPT: 'dog' ---
--- LAYER: 31 ---
================================================================================
Perturbing along first orthogonal direction (PC143)...

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
Original model completion: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.

--- Perturbing final token activation along orthogonal direction ---
Perturbation scale -100.0x: 건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건
Perturbation scale -20.0x: 건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건건
Perturbation scale -10.0x: A dog is a domesticated mammal belonging to the canine family, characterized by a furry or smooth coat, a dog's distinct barking sound, a generally omnivorous diet, a highly developed sense of smell and a highly social instinct.
Perturbation scale -5.0x: A dog is a domesticated mammal belonging to the Canis lupus familiaris species, characterized by a furry body, a distinctive barking sound, a highly developed sense of smell, a pack mentality, and a diverse range of breeds and sizes.
Perturbation scale -2.5x: A dog is a domesticated mammal characterized by a furry body, a wagging tail, a distinctive barking sound, a highly developed sense of smell, and a loyal and social nature.
Perturbation scale -1.5x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, a distinctive barking sound, and a highly social and loyal nature, often serving as a companion animal to humans.
Perturbation scale +0.0x: A dog is a domesticated mammal characterized by its furry coat, four legs, a tail, and a distinctive set of teeth and jaw structure, typically weighing between 20-200 pounds and standing between 10-30 inches tall, with a highly variable range of breeds and sizes.
Perturbation scale +1.5x: A dog is a domesticated, carnivorous mammal characterized by its upright posture, four legs, a wagging tail, and a variety of breeds with distinct physical and behavioral traits.
Perturbation scale +2.5x: A dog is a domesticated, carnivorous mammal characterized by its upright stance, four legs, a wagging tail, and a range of physical and behavioral traits that vary among breeds, including loyalty, playfulness, and a strong sense of smell.
Perturbation scale +5.0x: A typical dog is characterized by having four legs, a wagging tail, pointy ears, and teeth, with characteristics that vary among breeds, including different coat textures, colors, and body shapes.
Perturbation scale +10.0x: Dogs (Canidae family) exhibit key characteristics including: having teeth, teeth structure including carnassa teeth; having teeth structure including cheek teeth; having teeth structure including teeth in line; having teeth structure including teeth in line; having teeth structure including teeth in line; having teeth structure including teeth in line; having teeth structure including teeth in line; having
Perturbation scale +20.0x: Dogs146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146146
Perturbation scale +100.0x: ruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruhruh
================================================================================

================================================================================
--- ABLATION EXPERIMENT ON CONCEPT: 'dog' ---
--- LAYER: 31 ---
================================================================================

System Prompt: 'You are a helpful assistant.'
User Prompt:   'Please describe the essential features of a dog in one sentence.'
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

Saved average eigenvalue plot to lastToken_dog_avg_eigenvalue.png
Saved eigenvector similarity matrix to lastToken_dog_pc0_similarity.png
(interp) [cscandelius@holygpu8a19102 July]$