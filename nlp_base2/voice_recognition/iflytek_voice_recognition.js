/**
 * 科大讯飞语音识别程序 (JavaScript版本)
 * 支持在线和离线语音识别，结合消防项目应用场景
 */

class FirefightingVoiceRecognition {
    constructor(appId, apiKey, apiSecret) {
        this.appId = appId;
        this.apiKey = apiKey;
        this.apiSecret = apiSecret;
        
        // 消防相关语音命令词典
        this.firefightingCommands = {
            "火警": ["火警", "火灾", "起火", "燃烧", "烟雾", "报警"],
            "救援": ["救援", "救人", "被困", "疏散", "逃生", "安全"],
            "设备": ["设备", "器材", "水枪", "泡沫", "梯子", "呼吸器"],
            "位置": ["位置", "地点", "楼层", "房间", "方向", "坐标"],
            "状态": ["状态", "情况", "程度", "严重", "紧急", "危险"],
            "行动": ["行动", "执行", "开始", "停止", "继续", "完成"]
        };
        
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recognitionCallback = null;
    }

    /**
     * 初始化语音识别器
     */
    async initializeRecognizers() {
        try {
            // 检查浏览器支持
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('浏览器不支持语音录制功能');
            }

            // 检查科大讯飞SDK是否加载
            if (typeof window.IFlytekVoiceSDK === 'undefined') {
                throw new Error('科大讯飞SDK未加载，请先引入SDK文件');
            }

            // 初始化在线识别器
            this.onlineRecognizer = new window.IFlytekVoiceSDK.OnlineRecognition({
                appId: this.appId,
                apiKey: this.apiKey,
                apiSecret: this.apiSecret
            });

            // 初始化离线识别器
            this.offlineRecognizer = new window.IFlytekVoiceSDK.OfflineRecognition({
                appId: this.appId,
                apiKey: this.apiKey,
                apiSecret: this.apiSecret
            });

            console.log('语音识别器初始化成功');
            return true;
        } catch (error) {
            console.error('识别器初始化失败:', error);
            return false;
        }
    }

    /**
     * 录制音频
     * @param {number} duration - 录制时长（秒）
     * @returns {Promise<Blob>} 音频Blob对象
     */
    async recordAudio(duration = 5) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });

            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            this.audioChunks = [];

            return new Promise((resolve, reject) => {
                this.mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.audioChunks.push(event.data);
                    }
                };

                this.mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                    stream.getTracks().forEach(track => track.stop());
                    resolve(audioBlob);
                };

                this.mediaRecorder.onerror = (error) => {
                    stream.getTracks().forEach(track => track.stop());
                    reject(error);
                };

                this.mediaRecorder.start();
                console.log(`开始录制音频，时长: ${duration}秒`);

                setTimeout(() => {
                    this.mediaRecorder.stop();
                }, duration * 1000);
            });
        } catch (error) {
            console.error('音频录制失败:', error);
            throw error;
        }
    }

    /**
     * 在线语音识别
     * @param {Blob} audioBlob - 音频Blob对象
     * @returns {Promise<Object>} 识别结果
     */
    async onlineRecognition(audioBlob) {
        if (!this.onlineRecognizer) {
            throw new Error('在线识别器未初始化');
        }

        try {
            console.log('开始在线语音识别...');
            const result = await this.onlineRecognizer.recognize(audioBlob);
            
            // 分析消防相关关键词
            const firefightingAnalysis = this.analyzeFirefightingContent(result.text || '');
            
            return {
                type: 'online',
                text: result.text || '',
                confidence: result.confidence || 0,
                firefightingAnalysis: firefightingAnalysis,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('在线识别失败:', error);
            throw error;
        }
    }

    /**
     * 离线语音识别
     * @param {Blob} audioBlob - 音频Blob对象
     * @returns {Promise<Object>} 识别结果
     */
    async offlineRecognition(audioBlob) {
        if (!this.offlineRecognizer) {
            throw new Error('离线识别器未初始化');
        }

        try {
            console.log('开始离线语音识别...');
            const result = await this.offlineRecognizer.recognize(audioBlob);
            
            // 分析消防相关关键词
            const firefightingAnalysis = this.analyzeFirefightingContent(result.text || '');
            
            return {
                type: 'offline',
                text: result.text || '',
                confidence: result.confidence || 0,
                firefightingAnalysis: firefightingAnalysis,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('离线识别失败:', error);
            throw error;
        }
    }

    /**
     * 分析消防相关内容
     * @param {string} text - 识别的文本
     * @returns {Object} 分析结果
     */
    analyzeFirefightingContent(text) {
        const analysis = {
            categories: [],
            keywords: [],
            urgencyLevel: 'normal',
            actionRequired: false
        };

        const textLower = text.toLowerCase();

        // 检查消防类别
        for (const [category, keywords] of Object.entries(this.firefightingCommands)) {
            for (const keyword of keywords) {
                if (textLower.includes(keyword)) {
                    if (!analysis.categories.includes(category)) {
                        analysis.categories.push(category);
                    }
                    if (!analysis.keywords.includes(keyword)) {
                        analysis.keywords.push(keyword);
                    }
                }
            }
        }

        // 判断紧急程度
        const urgentKeywords = ["火警", "火灾", "起火", "燃烧", "被困", "危险", "紧急"];
        if (urgentKeywords.some(keyword => textLower.includes(keyword))) {
            analysis.urgencyLevel = 'urgent';
            analysis.actionRequired = true;
        }

        return analysis;
    }

    /**
     * 实时语音识别
     * @param {Function} callback - 识别结果回调函数
     * @param {number} duration - 识别时长（秒）
     */
    async realTimeRecognition(callback, duration = 10) {
        this.isRecording = true;
        this.recognitionCallback = callback;
        
        const startTime = Date.now();
        
        while (this.isRecording && (Date.now() - startTime) < duration * 1000) {
            try {
                // 录制短音频片段
                const audioBlob = await this.recordAudio(2);
                
                // 在线识别
                const result = await this.onlineRecognition(audioBlob);
                if (result.text) {
                    callback(result);
                }
                
                // 短暂休息
                await new Promise(resolve => setTimeout(resolve, 500));
            } catch (error) {
                console.error('实时识别错误:', error);
            }
        }
    }

    /**
     * 停止实时识别
     */
    stopRealTimeRecognition() {
        this.isRecording = false;
    }

    /**
     * 批量语音识别
     * @param {Array<Blob>} audioBlobs - 音频Blob数组
     * @param {boolean} useOffline - 是否使用离线识别
     * @returns {Promise<Array>} 识别结果数组
     */
    async batchRecognition(audioBlobs, useOffline = false) {
        const results = [];
        
        for (let i = 0; i < audioBlobs.length; i++) {
            try {
                let result;
                if (useOffline) {
                    result = await this.offlineRecognition(audioBlobs[i]);
                } else {
                    result = await this.onlineRecognition(audioBlobs[i]);
                }
                
                results.push({
                    index: i,
                    result: result
                });
            } catch (error) {
                results.push({
                    index: i,
                    error: error.message
                });
            }
        }
        
        return results;
    }

    /**
     * 创建音频播放器
     * @param {Blob} audioBlob - 音频Blob对象
     * @returns {HTMLAudioElement} 音频元素
     */
    createAudioPlayer(audioBlob) {
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        return audio;
    }

    /**
     * 下载音频文件
     * @param {Blob} audioBlob - 音频Blob对象
     * @param {string} filename - 文件名
     */
    downloadAudio(audioBlob, filename = 'firefighting_audio.webm') {
        const url = URL.createObjectURL(audioBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

/**
 * 消防语音识别演示类
 */
class FirefightingVoiceDemo {
    constructor() {
        this.recognizer = null;
        this.isInitialized = false;
    }

    /**
     * 初始化演示
     */
    async initialize() {
        // 配置信息（需要替换为实际的科大讯飞配置）
        const config = {
            appId: 'your_app_id_here',
            apiKey: 'your_api_key_here',
            apiSecret: 'your_api_secret_here'
        };

        this.recognizer = new FirefightingVoiceRecognition(
            config.appId,
            config.apiKey,
            config.apiSecret
        );

        this.isInitialized = await this.recognizer.initializeRecognizers();
        
        if (this.isInitialized) {
            console.log('✅ 语音识别器初始化成功');
            this.createUI();
        } else {
            console.error('❌ 识别器初始化失败，请检查配置');
        }
    }

    /**
     * 创建用户界面
     */
    createUI() {
        const container = document.getElementById('voice-recognition-demo') || document.body;
        
        container.innerHTML = `
            <div class="voice-recognition-container">
                <h2>🔥 消防项目语音识别演示</h2>
                <div class="controls">
                    <button id="record-btn" class="btn btn-primary">录制并识别</button>
                    <button id="online-btn" class="btn btn-success">在线识别</button>
                    <button id="offline-btn" class="btn btn-warning">离线识别</button>
                    <button id="realtime-btn" class="btn btn-info">实时识别</button>
                    <button id="stop-btn" class="btn btn-danger" style="display:none">停止识别</button>
                </div>
                <div class="results">
                    <h3>识别结果:</h3>
                    <div id="recognition-results"></div>
                </div>
                <div class="firefighting-analysis">
                    <h3>消防分析:</h3>
                    <div id="firefighting-results"></div>
                </div>
            </div>
            <style>
                .voice-recognition-container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                }
                .controls {
                    margin: 20px 0;
                }
                .btn {
                    padding: 10px 20px;
                    margin: 5px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                }
                .btn-primary { background-color: #007bff; color: white; }
                .btn-success { background-color: #28a745; color: white; }
                .btn-warning { background-color: #ffc107; color: black; }
                .btn-info { background-color: #17a2b8; color: white; }
                .btn-danger { background-color: #dc3545; color: white; }
                .results, .firefighting-analysis {
                    margin: 20px 0;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #f8f9fa;
                }
                .urgent {
                    background-color: #f8d7da;
                    border-color: #f5c6cb;
                    color: #721c24;
                }
            </style>
        `;

        this.bindEvents();
    }

    /**
     * 绑定事件
     */
    bindEvents() {
        document.getElementById('record-btn').addEventListener('click', () => this.recordAndRecognize());
        document.getElementById('online-btn').addEventListener('click', () => this.onlineRecognition());
        document.getElementById('offline-btn').addEventListener('click', () => this.offlineRecognition());
        document.getElementById('realtime-btn').addEventListener('click', () => this.realTimeRecognition());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopRecognition());
    }

    /**
     * 录制并识别
     */
    async recordAndRecognize() {
        try {
            this.showStatus('开始录制语音（5秒）...');
            const audioBlob = await this.recognizer.recordAudio(5);
            
            this.showStatus('进行在线识别...');
            const onlineResult = await this.recognizer.onlineRecognition(audioBlob);
            this.displayResults(onlineResult);
            
            this.showStatus('进行离线识别...');
            const offlineResult = await this.recognizer.offlineRecognition(audioBlob);
            this.displayResults(offlineResult, '离线识别结果');
            
        } catch (error) {
            this.showStatus(`错误: ${error.message}`);
        }
    }

    /**
     * 在线识别
     */
    async onlineRecognition() {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'audio/*';
        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    this.showStatus('进行在线识别...');
                    const result = await this.recognizer.onlineRecognition(file);
                    this.displayResults(result);
                } catch (error) {
                    this.showStatus(`错误: ${error.message}`);
                }
            }
        };
        fileInput.click();
    }

    /**
     * 离线识别
     */
    async offlineRecognition() {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'audio/*';
        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    this.showStatus('进行离线识别...');
                    const result = await this.recognizer.offlineRecognition(file);
                    this.displayResults(result);
                } catch (error) {
                    this.showStatus(`错误: ${error.message}`);
                }
            }
        };
        fileInput.click();
    }

    /**
     * 实时识别
     */
    async realTimeRecognition() {
        document.getElementById('realtime-btn').style.display = 'none';
        document.getElementById('stop-btn').style.display = 'inline-block';
        
        this.showStatus('开始实时语音识别（10秒）... 请说话...');
        
        const callback = (result) => {
            this.displayResults(result);
            if (result.firefightingAnalysis && result.firefightingAnalysis.actionRequired) {
                this.showStatus('🚨 检测到紧急情况，需要立即行动！', 'urgent');
            }
        };
        
        await this.recognizer.realTimeRecognition(callback, 10);
        this.stopRecognition();
    }

    /**
     * 停止识别
     */
    stopRecognition() {
        this.recognizer.stopRealTimeRecognition();
        document.getElementById('realtime-btn').style.display = 'inline-block';
        document.getElementById('stop-btn').style.display = 'none';
        this.showStatus('识别已停止');
    }

    /**
     * 显示状态
     */
    showStatus(message, className = '') {
        const resultsDiv = document.getElementById('recognition-results');
        resultsDiv.innerHTML = `<div class="${className}">${message}</div>`;
    }

    /**
     * 显示结果
     */
    displayResults(result, title = '识别结果') {
        const resultsDiv = document.getElementById('recognition-results');
        const firefightingDiv = document.getElementById('firefighting-results');
        
        resultsDiv.innerHTML = `
            <h4>${title}</h4>
            <p><strong>文本:</strong> ${result.text || '无'}</p>
            <p><strong>置信度:</strong> ${result.confidence || 0}</p>
            <p><strong>时间:</strong> ${result.timestamp || new Date().toISOString()}</p>
        `;
        
        if (result.firefightingAnalysis) {
            const analysis = result.firefightingAnalysis;
            firefightingDiv.innerHTML = `
                <p><strong>类别:</strong> ${analysis.categories.join(', ') || '无'}</p>
                <p><strong>关键词:</strong> ${analysis.keywords.join(', ') || '无'}</p>
                <p><strong>紧急程度:</strong> ${analysis.urgencyLevel}</p>
                <p><strong>需要行动:</strong> ${analysis.actionRequired ? '是' : '否'}</p>
            `;
        }
    }
}

// 页面加载完成后初始化演示
document.addEventListener('DOMContentLoaded', () => {
    const demo = new FirefightingVoiceDemo();
    demo.initialize();
});

// 导出类供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FirefightingVoiceRecognition, FirefightingVoiceDemo };
}



