import numpy as np
import matplotlib.pyplot as plt
import os

from rnn.tensor import tensor
from rnn.functions import Linear, Tanh
from rnn.loss import MSELoss
from rnn.optimizer import Adam

from data_gen_by_gesture.data_reader import read_one_circle_data


# =========================================================
# 0. create image folder
# =========================================================

os.makedirs("images", exist_ok=True)


# =========================================================
# 1. generate teacher data
# =========================================================

def generate_teacher_like_data():

    offsets = [(0.01,0.0),(0.0,0.0),(-0.01,0.01)]

    base_raw = read_one_circle_data()

    base = np.array(
        [[p["x"],p["y"]] for p in base_raw],
        dtype=np.float32
    )

    sequences=[]

    for dx,dy in offsets:

        shifted = base + np.array([[dx,dy]],dtype=np.float32)

        sequences.append(shifted)

    return sequences


raw_seqs = generate_teacher_like_data()

num_sequences = len(raw_seqs)


# =========================================================
# 2. convert to framework tensor format
# =========================================================

all_inputs=[]
all_targets=[]

for seq in raw_seqs:

    inp_seq=[]
    tgt_seq=[]

    for t in range(len(seq)-1):

        inp_seq.append(
            tensor(seq[t:t+1])
        )

        tgt_seq.append(
            tensor(seq[t+1:t+2])
        )

    all_inputs.append(inp_seq)
    all_targets.append(tgt_seq)


# =========================================================
# 3. RNN model definition (FORWARD RESTORED HERE)
# =========================================================

class Simple2DRNN:

    def __init__(self):

        self.ih = Linear(2,2)
        self.hh = Linear(2,2)
        self.act = Tanh()
        self.fc = Linear(2,2)


    def forward(self, x_seq, h0):

        h = h0

        outputs=[]

        for x in x_seq:

            h = self.act(
                self.ih(x) + self.hh(h)
            )

            y = self.fc(h)

            outputs.append(y)

        return outputs, h


    def __call__(self, x_seq, h0):

        return self.forward(x_seq, h0)


    def parameters(self):

        params=[]

        for layer in [self.ih,self.hh,self.fc]:

            params.append(layer.weight)

            if layer.bias is not None:
                params.append(layer.bias)

        return params


model = Simple2DRNN()


# =========================================================
# 4. initialization (match PyTorch)
# =========================================================

def xavier_(W):

    fan_in, fan_out = W.shape

    limit = np.sqrt(6.0/(fan_in+fan_out))

    W[:] = np.random.uniform(-limit,limit,size=W.shape)


def bias_init_(b):

    fan_in = b.shape[1]

    limit = 1/np.sqrt(fan_in)

    b[:] = np.random.uniform(-limit,limit,size=b.shape)


xavier_(model.ih.weight.data)
xavier_(model.hh.weight.data)
xavier_(model.fc.weight.data)

bias_init_(model.ih.bias.data)
bias_init_(model.hh.bias.data)
bias_init_(model.fc.bias.data)


# =========================================================
# 5. learnable initial hidden states
# =========================================================

c0_list=[

    tensor(
        np.random.randn(1,2)*0.1
    )

    for _ in range(num_sequences)
]


# =========================================================
# 6. optimizer and loss
# =========================================================

optimizer = Adam(
    model.parameters()+c0_list,
    lr=0.0005
)

criterion = MSELoss()


# =========================================================
# 7. training loop (PYTORCH-ALIGNED)
# =========================================================

print("Training RNN (framework version)...")

epochs=5000

loss_history=[]

for epoch in range(epochs):

    total_loss=0.0

    for i in range(num_sequences):

        outputs, h = model(
            all_inputs[i],
            c0_list[i]
        )

        T = len(outputs)

        loss = criterion(
            outputs[0],
            all_targets[i][0]
        )

        for t in range(1,T):

            loss = loss + criterion(
                outputs[t],
                all_targets[i][t]
            )

        loss = loss * tensor(1.0/T)


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.data


    avg_loss = total_loss/num_sequences

    loss_history.append(avg_loss)


    if (epoch+1)%200==0:

        print(
            f"Epoch {epoch+1}, Avg Loss {avg_loss:.6f}"
        )


# =========================================================
# 8. rollout
# =========================================================

def rollout(model,c0,x0,steps=1000):

    h=c0
    x=x0

    preds=[]
    states=[]

    preds.append(x.data.squeeze())
    states.append(h.data.squeeze())

    for _ in range(steps):

        h = model.act(
            model.ih(x) + model.hh(h)
        )

        x = model.fc(h)

        preds.append(x.data.squeeze())
        states.append(h.data.squeeze())

    return np.array(preds),np.array(states)


# =========================================================
# 9. arrow draw
# =========================================================

def draw_direction_arrow(ax,data,color,interval=30):

    for i in range(0,len(data)-1,interval):

        ax.annotate(
            '',
            xy=data[i+1],
            xytext=data[i],
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=1.5
            )
        )


# =========================================================
# 10. plot results
# =========================================================

plt.figure(figsize=(18,5))

ax0=plt.subplot(1,3,1)
ax1=plt.subplot(1,3,2)
ax2=plt.subplot(1,3,3)

colors=['purple','blue','green']


for i in range(num_sequences):

    teacher = raw_seqs[i]

    preds,states = rollout(
        model,
        c0_list[i],
        all_inputs[i][0]
    )

    ax0.plot(teacher[:,0],teacher[:,1],color=colors[i])
    draw_direction_arrow(ax0,teacher,colors[i])

    ax1.plot(preds[:,0],preds[:,1],color=colors[i])
    draw_direction_arrow(ax1,preds,colors[i])

    ax2.plot(states[:,0],states[:,1],color=colors[i])
    draw_direction_arrow(ax2,states,colors[i])


ax0.set_title("Teacher")
ax1.set_title("Learned Output")
ax2.set_title("Hidden State")

for ax in [ax0,ax1,ax2]:

    ax.axis("equal")
    ax.grid(True)


plt.tight_layout()

plt.savefig("images/framework_result.png",dpi=300)


# =========================================================
# 11. loss curve
# =========================================================

plt.figure()

plt.plot(loss_history)

plt.title("Loss")

plt.grid(True)

plt.savefig("images/framework_loss.png",dpi=300)