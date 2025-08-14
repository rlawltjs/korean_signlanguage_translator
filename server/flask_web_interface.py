#!/usr/bin/env python3
"""
Flask 웹 인터페이스 - 포즈 추정 및 수어 인식
"""

from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import time
import json
import os
from enhanced_webcam_client import WebcamCapture

app = Flask(__name__)
webcam = WebcamCapture()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """비디오 스트림"""
    return Response(webcam.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<action>')
def control(action):
    """웹캠 제어"""
    result = {'status': 'error', 'message': 'Unknown action'}
    
    if action == 'start_pose':
        result = {'status': 'success', 'message': webcam.start_realtime()}
    elif action == 'stop_pose':
        result = {'status': 'success', 'message': webcam.stop_realtime()}
    elif action == 'toggle_skeleton':
        result = {'status': 'success', 'message': webcam.toggle_skeleton_display()}
    elif action == 'toggle_sign_mode':
        result = {'status': 'success', 'message': webcam.toggle_sign_recognition()}
    elif action == 'toggle_sign_display':
        result = {'status': 'success', 'message': webcam.toggle_sign_prediction_display()}
    elif action == 'toggle_sign_smoothing':
        result = {'status': 'success', 'message': webcam.toggle_sign_smoothing()}
    elif action == 'clear_sign_buffer':
        webcam.clear_sign_buffer()
        result = {'status': 'success', 'message': '수어 버퍼 클리어됨'}
    elif action == 'start_recording':
        result = {'status': 'success', 'message': webcam.start_recording()}
    elif action == 'stop_recording':
        result = {'status': 'success', 'message': webcam.stop_recording()}
    elif action == 'start_capture':
        result = {'status': 'success', 'message': webcam.start_capture_images()}
    elif action == 'stop_capture':
        result = {'status': 'success', 'message': webcam.stop_capture_images()}
    
    return jsonify(result)

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """임계값 설정"""
    data = request.json
    threshold_type = data.get('type')
    value = float(data.get('value', 0))
    
    if threshold_type == 'pose':
        message = webcam.set_pose_threshold(value)
    elif threshold_type == 'sign':
        message = webcam.set_sign_confidence_threshold(value)
    else:
        return jsonify({'status': 'error', 'message': 'Invalid threshold type'})
    
    return jsonify({'status': 'success', 'message': message})

@app.route('/get_stats')
def get_stats():
    """서버 통계 조회"""
    stats = webcam.get_server_stats()
    if stats:
        return jsonify({'status': 'success', 'data': stats})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to get server stats'})

@app.route('/get_status')
def get_status():
    """현재 상태 조회"""
    status = {
        'pose_estimation': webcam.realtime_translate,
        'sign_recognition': webcam.sign_recognition_mode,
        'skeleton_display': webcam.show_skeleton,
        'sign_display': webcam.show_sign_prediction,
        'sign_smoothing': webcam.sign_smoothing_enabled,
        'recording': webcam.recording,
        'capturing': webcam.capturing_images,
        'pose_threshold': webcam.pose_threshold,
        'sign_threshold': webcam.sign_confidence_threshold,
        'last_sign_prediction': webcam.last_sign_prediction,
        'last_sign_confidence': webcam.last_sign_confidence,
        'client_id': webcam.client_id
    }
    return jsonify({'status': 'success', 'data': status})

@app.route('/recognize_sign', methods=['POST'])
def recognize_sign():
    """수어 인식 요청 처리"""
    try:
        # 요청 데이터 검증
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Invalid request data - features required'
            }), 400
            
        features = data['features']
        confidence_threshold = float(data.get('confidence_threshold', 0.6))
        
        # 수어 인식 수행
        result = webcam.recognize_sign_sequence(features, confidence_threshold)
        
        # 결과 반환
        response = {
            'status': 'success',
            'predictions': result.get('predictions', []),
            'message': result.get('message', ''),
            'has_model': result.get('has_model', False)
        }
        
        # 터미널에 결과 출력
        if result.get('predictions'):
            print("\n🤟 수어 인식 결과:")
            print("-" * 40)
            for pred in result['predictions']:
                print(f"단어: {pred['word']:<15} - 신뢰도: {pred['confidence']*100:>6.2f}%")
            print("-" * 40)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ 수어 인식 요청 처리 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# WebcamCapture 클래스에 메서드 추가 필요
def recognize_sign_sequence(self, features, confidence_threshold=0.6):
    """수어 시퀀스 인식"""
    if not self.recognizer or not self.recognizer.sign_model:
        return {
            'predictions': [],
            'message': 'Sign recognition model not loaded',
            'has_model': False
        }
    
    try:
        return self.recognizer.predict_sign_sequence(features, confidence_threshold)
    except Exception as e:
        print(f"❌ 수어 인식 실패: {e}")
        return {
            'predictions': [],
            'message': f'Recognition failed: {str(e)}',
            'has_model': True
        }
    
if __name__ == '__main__':
    # 템플릿 폴더 생성
    os.makedirs('templates', exist_ok=True)
    
    # HTML 템플릿 생성
    html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Webcam - Pose & Sign Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        .video-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .video-feed {
            max-width: 100%;
            border: 2px solid #333;
            border-radius: 8px;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .control-group {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .control-group h3 {
            margin-top: 0;
            color: #495057;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }
        .btn {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .btn.active {
            background-color: #28a745;
        }
        .btn.danger {
            background-color: #dc3545;
        }
        .btn.warning {
            background-color: #ffc107;
            color: #212529;
        }
        .slider-container {
            margin: 10px 0;
        }
        .slider-container label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .slider {
            width: 100%;
            margin: 5px 0;
        }
        .status {
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 12px;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .sign-prediction {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .stats-table th, .stats-table td {
            border: 1px solid #dee2e6;
            padding: 8px;
            text-align: left;
        }
        .stats-table th {
            background-color: #f8f9fa;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Enhanced Webcam</h1>
            <h2>포즈 추정 & 수어 인식 시스템</h2>
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" class="video-feed" alt="Webcam Feed">
        </div>
        
        <div class="controls">
            <!-- 포즈 추정 제어 -->
            <div class="control-group">
                <h3>🦴 포즈 추정</h3>
                <button class="btn" onclick="togglePoseEstimation()" id="poseBtn">포즈 추정 시작</button>
                <button class="btn" onclick="toggleSkeleton()" id="skeletonBtn">스켈레톤 표시</button>
                <div class="slider-container">
                    <label>포즈 임계값: <span id="poseThresholdValue">2.0</span></label>
                    <input type="range" class="slider" id="poseThreshold" min="1.0" max="5.0" step="0.1" value="2.0">
                </div>
            </div>
            
            <!-- 수어 인식 제어 -->
            <div class="control-group">
                <h3>🤟 수어 인식</h3>
                <button class="btn" onclick="toggleSignRecognition()" id="signModeBtn">수어 인식 시작</button>
                <button class="btn" onclick="toggleSignDisplay()" id="signDisplayBtn">수어 결과 표시</button>
                <button class="btn" onclick="toggleSignSmoothing()" id="signSmoothBtn">결과 평활화</button>
                <button class="btn warning" onclick="clearSignBuffer()">버퍼 클리어</button>
                <div class="slider-container">
                    <label>수어 신뢰도 임계값: <span id="signThresholdValue">0.6</span></label>
                    <input type="range" class="slider" id="signThreshold" min="0.1" max="1.0" step="0.1" value="0.6">
                </div>
                <div class="sign-prediction" id="signPrediction">
                    수어 인식 결과가 여기에 표시됩니다
                </div>
            </div>
            
            <!-- 녹화 및 캡처 -->
            <div class="control-group">
                <h3>📹 녹화 & 캡처</h3>
                <button class="btn" onclick="toggleRecording()" id="recordBtn">녹화 시작</button>
                <button class="btn" onclick="toggleCapture()" id="captureBtn">연속 캡처 시작</button>
                <button class="btn" onclick="captureImage()">단일 이미지 캡처</button>
            </div>
            
            <!-- 서버 정보 -->
            <div class="control-group">
                <h3>📊 서버 정보</h3>
                <button class="btn" onclick="refreshStats()">통계 새로고침</button>
                <div id="serverStats" class="status">
                    서버 통계를 가져오는 중...
                </div>
            </div>
        </div>
        
        <div id="statusMessage" class="status"></div>
    </div>

    <script>
        // 전역 상태 변수들
        let currentStatus = {
            pose_estimation: false,
            sign_recognition: false,
            skeleton_display: false,
            sign_display: false,
            sign_smoothing: true,
            recording: false,
            capturing: false
        };

        // 페이지 로드 시 상태 업데이트
        window.onload = function() {
            updateStatus();
            refreshStats();
            
            // 임계값 슬라이더 이벤트 리스너
            document.getElementById('poseThreshold').addEventListener('input', function() {
                const value = this.value;
                document.getElementById('poseThresholdValue').textContent = value;
                setThreshold('pose', value);
            });
            
            document.getElementById('signThreshold').addEventListener('input', function() {
                const value = this.value;
                document.getElementById('signThresholdValue').textContent = value;
                setThreshold('sign', value);
            });
            
            // 주기적 상태 업데이트 (3초마다)
            setInterval(updateStatus, 3000);
        };

        function showMessage(message, type = 'success') {
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            setTimeout(() => {
                statusDiv.textContent = '';
                statusDiv.className = 'status';
            }, 3000);
        }

        function updateButtonState(btnId, isActive, activeText, inactiveText) {
            const btn = document.getElementById(btnId);
            if (isActive) {
                btn.classList.add('active');
                btn.textContent = activeText;
            } else {
                btn.classList.remove('active');
                btn.textContent = inactiveText;
            }
        }

        async function controlAction(action) {
            try {
                const response = await fetch(`/control/${action}`);
                const result = await response.json();
                showMessage(result.message, result.status === 'success' ? 'success' : 'error');
                setTimeout(updateStatus, 500);
                return result.status === 'success';
            } catch (error) {
                showMessage(`오류: ${error.message}`, 'error');
                return false;
            }
        }

        async function setThreshold(type, value) {
            try {
                const response = await fetch('/set_threshold', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({type: type, value: value})
                });
                const result = await response.json();
                if (result.status !== 'success') {
                    showMessage(result.message, 'error');
                }
            } catch (error) {
                showMessage(`임계값 설정 오류: ${error.message}`, 'error');
            }
        }

        async function updateStatus() {
            try {
                const response = await fetch('/get_status');
                const result = await response.json();
                if (result.status === 'success') {
                    currentStatus = result.data;
                    
                    // 버튼 상태 업데이트
                    updateButtonState('poseBtn', currentStatus.pose_estimation, '포즈 추정 중지', '포즈 추정 시작');
                    updateButtonState('skeletonBtn', currentStatus.skeleton_display, '스켈레톤 숨기기', '스켈레톤 표시');
                    updateButtonState('signModeBtn', currentStatus.sign_recognition, '수어 인식 중지', '수어 인식 시작');
                    updateButtonState('signDisplayBtn', currentStatus.sign_display, '수어 결과 숨기기', '수어 결과 표시');
                    updateButtonState('signSmoothBtn', currentStatus.sign_smoothing, '평활화 끄기', '평활화 켜기');
                    updateButtonState('recordBtn', currentStatus.recording, '녹화 중지', '녹화 시작');
                    updateButtonState('captureBtn', currentStatus.capturing, '캡처 중지', '연속 캡처 시작');
                    
                    // 임계값 슬라이더 업데이트
                    document.getElementById('poseThreshold').value = currentStatus.pose_threshold;
                    document.getElementById('poseThresholdValue').textContent = currentStatus.pose_threshold;
                    document.getElementById('signThreshold').value = currentStatus.sign_threshold;
                    document.getElementById('signThresholdValue').textContent = currentStatus.sign_threshold;
                    
                    // 수어 예측 결과 표시
                    const signDiv = document.getElementById('signPrediction');
                    if (currentStatus.last_sign_prediction) {
                        signDiv.innerHTML = `
                            <strong>${currentStatus.last_sign_prediction}</strong><br>
                            <small>신뢰도: ${(currentStatus.last_sign_confidence * 100).toFixed(1)}%</small>
                        `;
                        signDiv.style.backgroundColor = currentStatus.last_sign_confidence > 0.7 ? '#d4edda' : '#fff3cd';
                    } else {
                        signDiv.textContent = '수어 인식 결과가 여기에 표시됩니다';
                        signDiv.style.backgroundColor = '#fff3cd';
                    }
                }
            } catch (error) {
                console.error('상태 업데이트 오류:', error);
            }
        }

        async function refreshStats() {
            try {
                const response = await fetch('/get_stats');
                const result = await response.json();
                if (result.status === 'success') {
                    const stats = result.data;
                    const statsDiv = document.getElementById('serverStats');
                    statsDiv.innerHTML = `
                        <table class="stats-table">
                            <tr><th>항목</th><th>값</th></tr>
                            <tr><td>포즈 추정 요청</td><td>${stats.request_count || 0}</td></tr>
                            <tr><td>수어 인식 요청</td><td>${stats.sign_request_count || 0}</td></tr>
                            <tr><td>수어 모델 로드됨</td><td>${stats.sign_model_loaded ? '✅' : '❌'}</td></tr>
                            <tr><td>활성 클라이언트</td><td>${stats.active_clients || 0}</td></tr>
                            <tr><td>평균 처리시간</td><td>${((stats.processing_times?.mean || 0) * 1000).toFixed(1)}ms</td></tr>
                            <tr><td>평균 수어 처리시간</td><td>${((stats.sign_processing_times?.mean || 0) * 1000).toFixed(1)}ms</td></tr>
                            <tr><td>검출 디바이스</td><td>${stats.detection_device || 'unknown'}</td></tr>
                            <tr><td>포즈 디바이스</td><td>${stats.pose_device || 'unknown'}</td></tr>
                        </table>
                    `;
                } else {
                    document.getElementById('serverStats').textContent = '서버 통계를 가져올 수 없습니다.';
                }
            } catch (error) {
                document.getElementById('serverStats').textContent = `통계 조회 오류: ${error.message}`;
            }
        }

        // 제어 함수들
        function togglePoseEstimation() {
            const action = currentStatus.pose_estimation ? 'stop_pose' : 'start_pose';
            controlAction(action);
        }

        function toggleSkeleton() {
            controlAction('toggle_skeleton');
        }

        function toggleSignRecognition() {
            controlAction('toggle_sign_mode');
        }

        function toggleSignDisplay() {
            controlAction('toggle_sign_display');
        }

        function toggleSignSmoothing() {
            controlAction('toggle_sign_smoothing');
        }

        function clearSignBuffer() {
            controlAction('clear_sign_buffer');
        }

        function toggleRecording() {
            const action = currentStatus.recording ? 'stop_recording' : 'start_recording';
            controlAction(action);
        }

        function toggleCapture() {
            const action = currentStatus.capturing ? 'stop_capture' : 'start_capture';
            controlAction(action);
        }

        function captureImage() {
            // 단일 이미지 캡처는 서버 구현이 필요하므로 일단 메시지만 표시
            showMessage('단일 이미지 캡처 기능은 추후 구현 예정입니다.', 'warning');
        }
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("🌐 Flask 웹 서버 시작")
    print("   - 주소: http://localhost:8000")
    print("   - 브라우저에서 접속하여 웹캠을 제어하세요")
    
    app.run(host='0.0.0.0', port=8000, debug=False)