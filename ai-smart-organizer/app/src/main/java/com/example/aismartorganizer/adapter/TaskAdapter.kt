package com.example.aismartorganizer.adapter

import android.graphics.Color
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.aismartorganizer.data.TaskEntity
import com.example.aismartorganizer.databinding.ItemTaskBinding

class TaskAdapter(private val onChecked: (TaskEntity, Boolean) -> Unit) : RecyclerView.Adapter<TaskAdapter.TaskVH>() {
    private var items: List<TaskEntity> = emptyList()

    fun submitList(list: List<TaskEntity>) {
        items = list
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): TaskVH {
        return TaskVH(ItemTaskBinding.inflate(LayoutInflater.from(parent.context), parent, false))
    }

    override fun onBindViewHolder(holder: TaskVH, position: Int) = holder.bind(items[position])
    override fun getItemCount(): Int = items.size

    inner class TaskVH(private val binding: ItemTaskBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(task: TaskEntity) {
            binding.cbTask.text = task.title
            binding.cbTask.isChecked = task.completed
            binding.priority.text = task.priority
            binding.priority.setTextColor(
                when (task.priority) {
                    "High" -> Color.parseColor("#EF4444")
                    "Medium" -> Color.parseColor("#F59E0B")
                    else -> Color.parseColor("#10B981")
                }
            )
            binding.cbTask.setOnCheckedChangeListener { _, isChecked -> onChecked(task, isChecked) }
        }
    }
}
