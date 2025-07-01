#!/bin/bash

# Ollama 설치 및 설정 스크립트
echo "🦙 Ollama 설치 및 설정 시작"
echo "================================"

# 1. Ollama 설치
echo "📦 Ollama 설치 중..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama가 이미 설치되어 있습니다."
    ollama --version
else
    echo "⬇️ Ollama 다운로드 및 설치..."
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo "✅ Ollama 설치 완료!"
    else
        echo "❌ Ollama 설치 실패"
        exit 1
    fi
fi

# 2. Ollama 서비스 시작
echo ""
echo "🚀 Ollama 서비스 시작..."

# systemd가 있는 경우
if command -v systemctl &> /dev/null; then
    sudo systemctl start ollama
    sudo systemctl enable ollama
    echo "✅ Ollama 서비스가 시작되었습니다."
else
    # 백그라운드에서 Ollama 서버 시작
    echo "🔧 Ollama 서버를 백그라운드에서 시작합니다..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 3
    
    if pgrep -f "ollama serve" > /dev/null; then
        echo "✅ Ollama 서버가 백그라운드에서 실행 중입니다."
    else
        echo "❌ Ollama 서버 시작 실패"
        exit 1
    fi
fi

# 3. 기본 모델 다운로드 (선택사항)
echo ""
echo "📥 기본 Gemma 모델 다운로드 (선택사항)..."
read -p "Gemma 2b 모델을 다운로드하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⬇️ Gemma 2b 모델 다운로드 중..."
    ollama pull gemma:2b
    
    if [ $? -eq 0 ]; then
        echo "✅ Gemma 2b 모델 다운로드 완료!"
    else
        echo "⚠️ Gemma 2b 모델 다운로드 실패 (나중에 다시 시도하세요)"
    fi
fi

# 4. 환경 변수 설정
echo ""
echo "🔧 환경 변수 설정..."

# .bashrc에 환경 변수 추가
if ! grep -q "OLLAMA_HOST" ~/.bashrc; then
    echo 'export OLLAMA_HOST=0.0.0.0:11434' >> ~/.bashrc
    echo "✅ OLLAMA_HOST 환경 변수 추가됨"
fi

if ! grep -q "OLLAMA_MODELS" ~/.bashrc; then
    echo 'export OLLAMA_MODELS=~/.ollama/models' >> ~/.bashrc
    echo "✅ OLLAMA_MODELS 환경 변수 추가됨"
fi

# 5. 방화벽 설정 (필요한 경우)
echo ""
echo "🔥 방화벽 설정 확인..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 11434/tcp
    echo "✅ UFW 방화벽에서 포트 11434 허용"
elif command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --permanent --add-port=11434/tcp
    sudo firewall-cmd --reload
    echo "✅ firewalld에서 포트 11434 허용"
else
    echo "⚠️ 방화벽 도구를 찾을 수 없습니다. 수동으로 포트 11434를 허용해주세요."
fi

# 6. 설치 확인
echo ""
echo "🧪 설치 확인..."
sleep 2

if ollama list &> /dev/null; then
    echo "✅ Ollama가 정상적으로 작동합니다!"
    echo ""
    echo "📋 설치된 모델 목록:"
    ollama list
else
    echo "❌ Ollama 설치 확인 실패"
    echo "로그를 확인하세요: cat ollama.log"
fi

# 7. 사용법 안내
echo ""
echo "🎉 Ollama 설치 완료!"
echo "================================"
echo ""
echo "📖 사용법:"
echo "  • 모델 목록 확인: ollama list"
echo "  • 모델 실행: ollama run <model_name>"
echo "  • 모델 다운로드: ollama pull <model_name>"
echo "  • 서버 상태 확인: ollama ps"
echo ""
echo "🔧 파인튜닝된 모델 등록:"
echo "  python ollama_setup.py setup"
echo ""
echo "💬 채팅 시작:"
echo "  python ollama_chat.py"
echo ""
echo "🌐 웹 인터페이스 (선택사항):"
echo "  Docker로 Ollama WebUI 실행:"
echo "  docker run -d --network=host -v ollama:/root/.ollama -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name ollama-webui --restart always ghcr.io/open-webui/open-webui:main"

# 환경 변수 적용을 위한 안내
echo ""
echo "⚠️ 중요: 새로운 환경 변수를 적용하려면 다음 중 하나를 실행하세요:"
echo "  source ~/.bashrc"
echo "  또는 터미널을 재시작하세요." 