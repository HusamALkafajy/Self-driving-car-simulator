# Hiding logs from terminal
print("Setting Up")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Imports
from utils import *
from sklearn.model_selection import train_test_split

# Load data info
path = "Data"
data = import_data_info(path)

# Visualize and balance data
data = balance_data(data,display=False)

# Load data
img_paths, steerings =load_data(path,data)
#print(img_path[0],steering[0])

# Spliting data
X_train,X_test,y_train,y_test = train_test_split(img_paths,steerings,test_size=0.2,random_state=5)
print(f"Total trainning Images: {len(X_train)}")
print(f"Total testing Images: {len(X_test)}")


# Create model
model = create_model()
model.summary()

# Train model
history = model.fit(batch_gen(X_train,y_train,100,1),steps_per_epoch=300,epochs=10,validation_data=batch_gen(X_test,y_test,100,0),validation_steps=200)
# batch = 100 , steps_per_epoch = 300 and epochs =  10
# For validation batch = 100 and steps = 200

# Save model
model.save('model.h5')
print("Model Saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title("Loss")
plt.xlabel('Epoch')
plt.show()
plt.savefig()