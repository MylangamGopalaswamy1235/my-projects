package com.example.aismartorganizer.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.aismartorganizer.data.NoteEntity
import com.example.aismartorganizer.databinding.ItemTreeNodeBinding

class TreeAdapter : RecyclerView.Adapter<TreeAdapter.TreeVH>() {
    private var items: List<Pair<NoteEntity, Int>> = emptyList()

    fun submitTree(flattened: List<Pair<NoteEntity, Int>>) {
        items = flattened
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): TreeVH {
        return TreeVH(ItemTreeNodeBinding.inflate(LayoutInflater.from(parent.context), parent, false))
    }

    override fun onBindViewHolder(holder: TreeVH, position: Int) = holder.bind(items[position])
    override fun getItemCount(): Int = items.size

    inner class TreeVH(private val binding: ItemTreeNodeBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(node: Pair<NoteEntity, Int>) {
            binding.tvNode.text = node.first.title
            binding.indentSpace.layoutParams.width = node.second * 40
            binding.indentSpace.requestLayout()
            binding.connector.visibility = if (node.second > 0) View.VISIBLE else View.INVISIBLE
        }
    }
}
