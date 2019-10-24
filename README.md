# Data-extraction-from-SELFIE-images
Extract user information from the profile picture


The point of this project was extracting all the information possible from a user just using the profile picture and from there use it to star recommending stuff based on the age, gender, ...

The model uses the Selfie Dataset from the University of Central Florida (https://www.crcv.ucf.edu/data/Selfie/) which contains arround 
+46k images with information like the age interval, gender, color hair, facial expression, race, ...

To train the model I used a multioutput neural network using the Keras API. First, the image is process by a ConvNet (similar to the VGG16 one) and the a couple of dense layers on top which then separate into the different number of outputs.

The model archived very good accuracy in some of the features (age, race, hair color, ...) but very poorly in others (facial expresssion, gender, ...). Actually the gender reached a 75% of accuracy on the validation set but in the images I've personally picked of celebrities, almost always predicted female, I think is due to the ammount of girls on the image dataset being way bigger than men's.

Of course it was very fun to do and the results where pretty good considering this was my first multioutput network. It needs to balance the losses of the multiple features.
I would love to try to adjust some hyperparameters and do some more tests, but just using the 40% of dataset, my laptop (ThinkPad E590 default) took like 5h, so with university stuff to be done I cannot stop using the PC for test (although I would love to).
