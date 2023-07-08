from RTSR3_tf import *

rtsr3 = RTSR3(2,3)
rtsr3.build(input_shape=(1,180,390,3))
print(rtsr3.summary())

rtsr3.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

hisfea_pool = [tf.constant(np.zeros((1,180,390,32),dtype=np.float32)),tf.constant(np.zeros((1,180,390,32),dtype=np.float32))]

history = rtsr3.fit(generate_data("../data/540cut","../data/1080cut"), steps_per_epoch=100, epochs=50, verbose=1)