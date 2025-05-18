$(document).ready(function() {
    // アコーディオン機能
    $('.accordion-header').click(function() {
        $(this).toggleClass('active');
        $(this).next('.accordion-content').toggleClass('active');
    });
    
    // フォーム送信
    $('#transcription-form').submit(function(event) {
        event.preventDefault();
        
        // 送信前の検証
        const fileInput = document.getElementById('file');
        if (!fileInput.files[0]) {
            showAlert('ファイルを選択してください', 'error');
            return;
        }
        
        // アラート表示クリア
        $('#alerts').empty();
        
        // フォームデータ取得
        const formData = new FormData(this);
        
        // ファイルサイズ表示
        const fileSize = (fileInput.files[0].size / (1024 * 1024)).toFixed(2);
        
        // 処理状態表示
        $('#status').text(`ファイルをアップロード中...(${fileSize}MB)`);
        $('#progress-bar').width('10%');
        $('#progress-text').text('10%');
        
        // Ajax送信
        $.ajax({
            url: '/transcribe',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
                $('#status').text('文字起こし処理を開始しました');
                showAlert('処理を開始しました。完了までお待ちください。', 'success');
                checkStatus(data.task_id);
            },
            error: function(xhr) {
                let errorMessage = '処理開始中にエラーが発生しました';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage += ': ' + xhr.responseJSON.error;
                }
                $('#status').text('エラーが発生しました');
                $('#progress-bar').width('0%');
                $('#progress-text').text('');
                showAlert(errorMessage, 'error');
            }
        });
    });
    
    // 処理状態確認
    function checkStatus(taskId) {
        $.get('/status/' + taskId, function(data) {
            // 状態表示更新
            $('#status').text('状態: ' + getStatusText(data.status));
            
            // 進捗表示更新
            if (data.state === 'PROGRESS') {
                $('#progress-bar').width(data.progress + '%');
                $('#progress-text').text(data.progress + '%');
            }
            
            // 処理完了
            if (data.state === 'SUCCESS') {
                $('#progress-bar').width('100%');
                $('#progress-text').text('100%');
                $('#result').html('<p>文字起こしが完了しました！</p><a href="/result/' + 
                    taskId + '" target="_blank" class="result-link">テキストファイルをダウンロード</a>');
                showAlert('文字起こしが完了しました！ダウンロードリンクをクリックしてファイルを保存してください。', 'success');
            } 
            // 処理失敗
            else if (data.state === 'FAILURE') {
                $('#status').text('文字起こしに失敗しました');
                $('#progress-bar').width('0%');
                $('#progress-text').text('');
                showAlert('文字起こし処理に失敗しました: ' + data.error, 'error');
            } 
            // 処理継続中
            else {
                setTimeout(function() {
                    checkStatus(taskId);
                }, 3000);  // 3秒ごとに状態確認
            }
        }).fail(function() {
            $('#status').text('状態確認中にエラーが発生しました');
            showAlert('処理状態の取得に失敗しました。更新してやり直してください。', 'error');
        });
    }
    
    // 状態テキスト変換
    function getStatusText(status) {
        switch (status) {
            case 'Task is waiting for execution.':
                return '処理待機中...';
            case 'Task is in progress.':
                return '文字起こし処理中...';
            case 'Task completed successfully.':
                return '文字起こし完了！';
            case 'Task failed.':
                return '処理に失敗しました';
            default:
                return status;
        }
    }
    
    // アラート表示
    function showAlert(message, type) {
        const alertClass = type === 'error' ? 'alert-error' : 
                          type === 'warning' ? 'alert-warning' : 'alert-success';
        
        const alertHtml = `<div class="alert ${alertClass}">${message}</div>`;
        $('#alerts').html(alertHtml);
        
        // 成功メッセージは5秒後に消える
        if (type === 'success') {
            setTimeout(function() {
                $('#alerts .alert-success').fadeOut(500, function() {
                    $(this).remove();
                });
            }, 5000);
        }
    }
});