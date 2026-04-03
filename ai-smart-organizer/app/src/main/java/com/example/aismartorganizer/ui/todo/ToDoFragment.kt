package com.example.aismartorganizer.ui.todo

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.aismartorganizer.SmartOrganizerApp
import com.example.aismartorganizer.adapter.TaskAdapter
import com.example.aismartorganizer.databinding.FragmentTodoBinding
import com.example.aismartorganizer.utils.DataStructureUtils
import com.example.aismartorganizer.viewmodel.MainViewModel
import com.example.aismartorganizer.viewmodel.ViewModelFactory

class ToDoFragment : Fragment() {

    private var _binding: FragmentTodoBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MainViewModel by activityViewModels {
        ViewModelFactory((requireActivity().application as SmartOrganizerApp).repository)
    }

    private val adapter = TaskAdapter { task, checked -> viewModel.toggleTask(task, checked) }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentTodoBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.todoRecycler.layoutManager = LinearLayoutManager(requireContext())
        binding.todoRecycler.adapter = adapter

        viewModel.tasks.observe(viewLifecycleOwner) { tasks ->
            adapter.submitList(tasks)
            val queue = DataStructureUtils.prioritizeTasks(tasks)
            binding.queueOrder.text = queue.joinToString(" -> ") { "${it.title}(${it.priority})" }
        }

        binding.btnAddTask.setOnClickListener {
            val title = binding.inputTask.text.toString().ifBlank { "New Task" }
            val priority = binding.prioritySpinner.selectedItem.toString()
            viewModel.addTask(title, priority)
            binding.inputTask.text?.clear()
        }

        if (viewModel.tasks.value.isNullOrEmpty()) {
            viewModel.addTask("Revise DBMS", "High")
            viewModel.addTask("Prepare slides", "Medium")
            viewModel.addTask("Water plants", "Low")
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
