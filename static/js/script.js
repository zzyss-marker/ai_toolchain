$(document).ready(function() {
    // 获取组件列表
    $.get('/get_components', function(data) {
        var datasets = data.datasets;
        var modules = data.modules;
        var saved_models = data.saved_models;

        for (var i = 0; i < datasets.length; i++) {
            $('#datasets').append('<div class="draggable" data-type="dataset">' + datasets[i] + '</div>');
        }
        for (var i = 0; i < modules.length; i++) {
            $('#modules').append('<div class="draggable" data-type="module">' + modules[i] + '</div>');
        }
        for (var i = 0; i < saved_models.length; i++) {
            $('#saved-models').append('<div class="saved-model">' + saved_models[i] + '</div>');
        }

        // 可拖拽组件
        $('.draggable').draggable({
            helper: 'clone'
        });

        // 放置区域
        $('#drop-area').droppable({
            accept: '.draggable',
            drop: function(event, ui) {
                var type = ui.draggable.data('type');
                var name = ui.draggable.text();
                // 添加删除按钮和拖动手柄
                $(this).append(
                    `<div class="dropped-component" data-type="${type}">
                        <span class="drag-handle">⋮</span>
                        <span class="component-name">${name}</span>
                        <button class="remove-btn" title="删除">×</button>
                    </div>`
                );
                setTimeout(visualizeModel, 100);
            }
        });

        // 使放置的组件可排序
        $('#drop-area').sortable({
            handle: '.drag-handle',
            update: function(event, ui) {
                visualizeModel();
            }
        });

        // 删除组件
        $(document).on('click', '.remove-btn', function() {
            $(this).parent().remove();
            visualizeModel();
        });
    });

    // 添加模型结构可视化
    function visualizeModel() {
        var layers = [];
        $('#drop-area .dropped-component').each(function() {
            layers.push($(this).find('.component-name').text());
        });
        
        if (layers.length > 0) {
            var svg = d3.select("#model-visualization")
                .html("") // 清空现有内容
                .append("svg")
                .attr("width", 600)
                .attr("height", layers.length * 60);

            // 绘制每一层
            layers.forEach((layer, i) => {
                var g = svg.append("g")
                    .attr("transform", `translate(50, ${i * 60 + 30})`);
                
                // 绘制节点
                g.append("rect")
                    .attr("width", 500)
                    .attr("height", 40)
                    .attr("rx", 5)
                    .attr("ry", 5)
                    .attr("class", "layer-node");
                
                // 添加文字
                g.append("text")
                    .attr("x", 250)
                    .attr("y", 25)
                    .attr("text-anchor", "middle")
                    .text(layer);
                
                // 如果不是最后一层，添加连接线
                if (i < layers.length - 1) {
                    svg.append("path")
                        .attr("d", `M300,${i * 60 + 70} L300,${i * 60 + 90}`)
                        .attr("stroke", "#999")
                        .attr("stroke-width", 2)
                        .attr("marker-end", "url(#arrow)");
                }
            });
        }
    }

    // 监听组件拖放，实时更新可视化
    $('#drop-area').on('drop', function() {
        setTimeout(visualizeModel, 100); // 延迟一下等待DOM更新
    });

    // 修改训练按钮点击事件
    $('#train-button').click(function() {
        var components = $('#drop-area .dropped-component');
        if (components.length === 0) {
            alert('请先添加模型层！');
            return;
        }

        var model_config = [];
        components.each(function() {
            var type = $(this).data('type');
            var name = $(this).find('.component-name').text();
            var params = {}; // 可以添加参数配置功能
            model_config.push({'type': name, 'params': params});
        });

        var training_params = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 5
        };

        // 禁用训练按钮
        $('#train-button').prop('disabled', true).text('训练中...');
        
        // 重置进度条
        $('#training-progress .progress-bar')
            .css('width', '0%')
            .text('0%');
        $('#training-progress').show();

        // 开始轮询训练进度
        var progressCheck = setInterval(function() {
            $.get('/training_progress', function(data) {
                var progress = data.progress;
                $('#training-progress .progress-bar')
                    .css('width', progress + '%')
                    .text(progress + '%');
                
                if (progress >= 100) {
                    clearInterval(progressCheck);
                }
            });
        }, 1000);

        $.ajax({
            url: '/train',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({'model_config': model_config, 'training_params': training_params}),
            success: function(response) {
                clearInterval(progressCheck);
                if (response.status === 'success') {
                    $('#training-progress .progress-bar')
                        .css('width', '100%')
                        .text('100%');
                    
                    alert('训练完成！\n模型准确率: ' + response.accuracy.toFixed(2) + '%\n模型已保存为：' + response.model_name);
                    window.location.href = response.redirect_url;
                } else {
                    alert('训练失败：' + response.message);
                }
            },
            error: function(xhr) {
                clearInterval(progressCheck);
                alert('训练过程中出现错误：' + (xhr.responseJSON ? xhr.responseJSON.message : '未知错误'));
            },
            complete: function() {
                // 恢复训练按钮
                $('#train-button').prop('disabled', false).text('开始训练');
            }
        });
    });
});
