function press_btn_rock () {
    gameControl.play('Rock');
}
function press_btn_paper () {
    gameControl.play('Paper');
}
function press_btn_scissors () {
    gameControl.play('Scissors');
}

const Memory = {
    memorySize: 20,
    main: [],
    saveData: function (data) {
        // data = [input:array, label:array]
        if (this.main.length == this.memorySize) {
            this.main.splice(0,1);
        }
        this.main.push(data);
    },
    getMemoryTensor: function () {
        let inputArray = [],
        labelArray = [];
        for (let i = 0; i < this.main.length; i++) {
            inputArray.push(this.main[i][0])
            labelArray.push(this.main[i][1])
        }
        return {
            inputBatch: tf.tensor2d(inputArray),
            labelBatch: tf.tensor2d(labelArray)
        }
    }
}

const Model = {
    batchSize: 8,
    net: tf.sequential({
        layers: [
            tf.layers.dense({inputShape: [2*2*3], units: 3, activation: 'softmax'}),
        ]
    }),
    compile: function () {
        this.net.compile({
            optimizer: tf.train.sgd(0.1),
            shuffle: true,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    },
    predict: function (input) {
        // input array; return tensor1d
        return this.net.predict(tf.tensor2d([input])).squeeze();
    },
    update: async function (inputBatch, labelBatch) {
        // inputBatch = tf.tensor2d(memorySize × 12), labelBatch = tf.tensor2d(memorySize × 3)
        this.compile();
        await this.net.fit(inputBatch, labelBatch, {
            batchSize: this.batchSize,
            epochs: 2
        }).then(info => {
            console.log('Final accuracy', info.history.acc);
        });
    }
}

const Agent = {
    memory: Memory,
    model: Model,
    history: {
        historySize: 50,
        main: []
    },
    actionMap: {'Rock':[1,0,0], 'Scissors':[0,1,0], 'Paper':[0,0,1]},
    predict: function (input) {
        // input = array(12)
        return this.model.predict(input);
    },
    randomGetAction: function () {
        let actionKeys = Object.keys(this.actionMap);
        return actionKeys[Math.floor(Math.random()*actionKeys.length)];
    },
    getAction: function (isGreedy=false) {
        // If lack of history data, choose action randomly
        if (this.history.main.length < this.model.batchSize) {
            return this.randomGetAction();
        }
        // Predict human action based on last two history data
        const inputArray = this.history.main[this.history.main.length-2][0].concat(
            this.history.main[this.history.main.length-2][1],
            this.history.main[this.history.main.length-1][0],
            this.history.main[this.history.main.length-1][1],
        );
        const humamActionPredict = this.predict(inputArray); // humanActionPredict = tensor1d
        console.log('Human action prediction: ', humamActionPredict.arraySync());
        // Choose counter action
        let predAction;
        let actionKeys = Object.keys(this.actionMap);
        if (isGreedy == true) {
            let indexOfMax = humamActionPredict.argMax().arraySync();
            predAction = actionKeys[indexOfMax];
        } 
        else {
            let sampleIndex = tf.multinomial(humamActionPredict, 1, null, true).arraySync(); // choose action by distribution
            predAction = actionKeys[sampleIndex];
        }
        switch (predAction) {
            case 'Rock': return 'Paper';
            case 'Scissors': return 'Rock';
            case 'Paper': return 'Scissors';
        }
    },
    feedHistory: function (data) {
        // data = [humanAction = array(3), agentAction = array(3)]
        if (this.history.main.length == this.history.historySize) {
            this.history.main.splice(0,1);
        }
        this.history.main.push(data);
        this.feedMemory();
        this.updateModel();
    },
    feedMemory: function () {
        if (this.history.main.length >= 3) {
            const inputArray = this.history.main[this.history.main.length-3][0].concat(
                this.history.main[this.history.main.length-3][1],
                this.history.main[this.history.main.length-2][0],
                this.history.main[this.history.main.length-2][1]  
            );
            const labelArray = this.history.main[this.history.main.length-1][0];
            this.memory.saveData([inputArray, labelArray]);
        }
    },
    updateModel: async function () {
        if (this.memory.main.length >= this.model.batchSize) {
            console.log('Update model...');
            let {inputBatch, labelBatch} = this.memory.getMemoryTensor();
            this.model.update(inputBatch, labelBatch);
        }
    }    
}

const gameControl = {
    img: {'Rock':null, 'Paper':null, 'Scissors':null},
    btn: {'Rock':null, 'Paper':null, 'Scissors':null}, 
    agent: Agent,
    loadImage: function () {
        this.img['Rock'] = new Image();
        this.img['Rock'].src = 'image/Rock.png';
        this.img['Paper'] = new Image();
        this.img['Paper'].src = 'image/Paper.png';
        this.img['Scissors'] = new Image();
        this.img['Scissors'].src = 'image/Scissors.png';
    },
    loadBtn: function () {
        this.btn['Rock'] = document.getElementById('btn_rock');
        this.btn['Paper'] = document.getElementById('btn_paper');
        this.btn['Scissors'] = document.getElementById('btn_scissors');
    },
    disableBtn: function () {
        this.btn['Rock'].disabled = true;
        this.btn['Paper'].disabled = true;
        this.btn['Scissors'].disabled = true;
    },
    enableBtn: function () {
        this.btn['Rock'].disabled = false;
        this.btn['Paper'].disabled = false;
        this.btn['Scissors'].disabled = false;
    },
    play: function (humanAction) {
        let agentAction = this.agent.getAction();
        let data = [this.agent.actionMap[humanAction], this.agent.actionMap[agentAction]];
        this.agent.feedHistory(data);
        this.render(humanAction, agentAction);
    },
    render: function (humanAction, agentAction) {
        function drawImage(image, x, y, scale, rotation){
            ctx.save();
            ctx.setTransform(scale, 0, 0, scale, x, y); // sets scale and origin
            ctx.rotate(rotation);
            ctx.drawImage(image, -image.width / 2, -image.height / 2);
            ctx.restore();
        }
        let ctx = document.getElementById('canvas').getContext('2d');
        let canvas = document.getElementById('canvas');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawImage(this.img[agentAction], 100, 80, 0.225, Math.PI);
        drawImage(this.img[humanAction], 100, 220, 0.225, 0);
    }
}
gameControl.loadImage();
gameControl.loadBtn();
